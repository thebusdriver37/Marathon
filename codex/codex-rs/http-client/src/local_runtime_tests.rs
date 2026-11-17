use super::validate_local_runtime_url;
use pretty_assertions::assert_eq;

#[test]
fn local_runtime_accepts_only_literal_loopback_urls() {
    for url in [
        "http://127.0.0.1:18111/v1",
        "ws://127.0.0.1:18111/v1/responses",
        "http://[::1]:18111/",
        "https://127.0.0.2/",
    ] {
        assert_eq!(validate_local_runtime_url(url).is_ok(), true, "{url}");
    }
    for url in [
        "https://chatgpt.com/",
        "https://api.github.com/",
        "http://192.168.1.1/",
        "http://8.8.8.8/",
        "http://localhost/",
        "http://127.0.0.1.example/",
        "http://127.0.0.1@external.example/",
        "http://secret@127.0.0.1/",
        "file:///tmp/test",
        "not a URL",
        "http://[::ffff:8.8.8.8]/",
    ] {
        assert_eq!(
            validate_local_runtime_url(url).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied,
            "{url}"
        );
    }
}
