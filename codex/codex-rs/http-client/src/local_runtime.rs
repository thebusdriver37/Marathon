//! Marathon's frontend transport boundary. Tool subprocesses retain their own networking.

use std::io;
use std::net::IpAddr;

pub fn local_runtime_enabled() -> bool {
    std::env::var_os("MARATHON_LOCAL_ONLY").is_some_and(|value| value == "1")
}

/// Accept literal loopback destinations only, without DNS or proxy resolution.
pub fn validate_local_runtime_url(url: &str) -> io::Result<()> {
    let allowed = url.parse::<http::Uri>().ok().is_some_and(|uri| {
        matches!(uri.scheme_str(), Some("http" | "https" | "ws" | "wss"))
            && uri.host().is_some_and(|host| {
                host.trim_start_matches('[')
                    .trim_end_matches(']')
                    .parse::<IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
            })
            && uri
                .authority()
                .is_some_and(|authority| !authority.as_str().contains('@'))
    });
    if allowed {
        Ok(())
    } else {
        // Do not include URLs, which may contain credentials or private queries.
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Marathon local runtime blocked a non-loopback frontend destination",
        ))
    }
}

#[cfg(test)]
#[path = "local_runtime_tests.rs"]
mod tests;
