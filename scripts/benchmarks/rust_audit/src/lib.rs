//! Synthetic benchmark prototypes only. Not linked into Marathon.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{panic::catch_unwind, slice, str};

fn repeated<T: PartialEq>(tail: &[T]) -> bool {
    if tail.len() < 2048 {
        return false;
    }
    for width in 1..=128 {
        let repeats = 32.max((2048 + width - 1) / width);
        let span = width * repeats;
        if span <= tail.len() {
            let end = &tail[tail.len() - span..];
            let unit = &tail[tail.len() - width..];
            if end.chunks_exact(width).all(|part| part == unit) {
                return true;
            }
        }
    }
    false
}

fn repetition(text: &str) -> bool {
    if text.is_ascii() {
        repeated(&text.as_bytes()[text.len().saturating_sub(8192)..])
    } else {
        // Collect only the suffix, then restore its original order before testing
        // suffix spans. Widths are Unicode codepoints, as in Python.
        let mut tail = text.chars().rev().take(8192).collect::<Vec<_>>();
        tail.reverse();
        repeated(&tail)
    }
}

fn fingerprint(input: &[u8]) -> Value {
    let items: Vec<Value> = serde_json::from_slice(input).unwrap();
    let mut digest = Sha256::new();
    digest.update(b"[");
    let mut fingerprints = Vec::new();
    let mut bytes = 2;
    for (i, item) in items.iter().enumerate() {
        // Default serde_json maps sort keys. Float spelling is NOT Python-compatible;
        // the benchmark validates string/integer fixtures only.
        let encoded = serde_json::to_vec(item).unwrap();
        if i > 0 {
            digest.update(b",");
            bytes += 1;
        }
        digest.update(&encoded);
        bytes += encoded.len();
        if i < 256 {
            fingerprints.push(format!("{:x}", Sha256::digest(&encoded)));
        }
    }
    digest.update(b"]");
    json!({"count": items.len(), "bytes": bytes,
        "hash": format!("{:x}", digest.finalize()), "fingerprints": fingerprints})
}

fn parse_frame(frame: &[u8], output: &mut Vec<Value>) {
    let text = String::from_utf8_lossy(frame)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let data = text
        .split('\n')
        .filter_map(|line| {
            line.strip_prefix("data:")
                .map(|v| v.trim_start_matches(' '))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if let Ok(value @ Value::Object(_)) = serde_json::from_str(&data) {
        output.push(value);
    }
}

fn sse(input: &[u8], chunk_size: usize) -> Value {
    let mut buffer = Vec::new();
    let mut output = Vec::new();
    let mut scan = 0;
    for chunk in input.chunks(chunk_size) {
        buffer.extend_from_slice(chunk);
        let mut cursor = 0;
        loop {
            let remaining = &buffer[scan..];
            let lf = memchr::memmem::find(remaining, b"\n\n").map(|i| (i, 2));
            let crlf = memchr::memmem::find(remaining, b"\r\n\r\n").map(|i| (i, 4));
            let Some((position, width)) = lf.into_iter().chain(crlf).min() else {
                scan = cursor.max(buffer.len().saturating_sub(3));
                break;
            };
            let end = scan + position;
            parse_frame(&buffer[cursor..end], &mut output);
            cursor = end + width;
            scan = cursor;
        }
        buffer.drain(..cursor);
        scan -= cursor;
    }
    if !buffer.is_empty() {
        parse_frame(&buffer, &mut output);
    }
    Value::Array(output)
}

/// Caller supplies valid input/output pointers and capacities. Returns usize::MAX
/// on an invalid operation, parse failure, or undersized output. Audit harness only.
#[no_mangle]
pub unsafe extern "C" fn audit_json(
    op: u32,
    input: *const u8,
    len: usize,
    output: *mut u8,
    capacity: usize,
    chunk_size: usize,
) -> usize {
    catch_unwind(|| {
        let input = slice::from_raw_parts(input, len);
        let value = match op {
            1 => fingerprint(input),
            2 => sse(input, chunk_size.max(1)),
            _ => panic!("unknown audit operation"),
        };
        let encoded = serde_json::to_vec(&value).unwrap();
        assert!(encoded.len() <= capacity);
        slice::from_raw_parts_mut(output, capacity)[..encoded.len()].copy_from_slice(&encoded);
        encoded.len()
    })
    .unwrap_or(usize::MAX)
}

#[no_mangle]
pub unsafe extern "C" fn audit_repetition(input: *const u8, len: usize) -> i32 {
    catch_unwind(|| repetition(str::from_utf8(slice::from_raw_parts(input, len)).unwrap()) as i32)
        .unwrap_or(-1)
}
