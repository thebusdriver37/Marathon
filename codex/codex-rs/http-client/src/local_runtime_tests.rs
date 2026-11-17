use super::validate_local_runtime_url;
use pretty_assertions::assert_eq;

#[tokio::test]
async fn local_runtime_transport_blocks_cloud_and_redirects_without_proxying() {
    // Set environment only in a fresh process, never underneath other Tokio tests.
    if !super::local_runtime_enabled() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "local_runtime::tests::local_runtime_transport_blocks_cloud_and_redirects_without_proxying",
                "--nocapture",
            ])
            .env("MARATHON_LOCAL_ONLY", "1")
            .env("HTTP_PROXY", "http://192.0.2.1:9")
            .env("HTTPS_PROXY", "http://192.0.2.1:9")
            .env("ALL_PROXY", "http://192.0.2.1:9")
            .env("NO_PROXY", "")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        return;
    }
    let pool = crate::RouteAwareClientPool::new_without_request_logging(
        crate::HttpClientFactory::new(crate::OutboundProxyPolicy::ReqwestDefault),
        crate::ClientRouteClass::Api,
    );
    for url in [
        "https://api.openai.com/v1/responses",
        "https://chatgpt.com/backend-api/",
    ] {
        let error = pool.get(url).send().await.unwrap_err();
        assert!(error.to_string().contains("non-loopback"), "{error}");
    }

    use std::io::Read;
    use std::io::Write;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                        .unwrap();
                    let mut buffer = [0; 4096];
                    assert!(stream.read(&mut buffer).unwrap() > 0);
                    stream.write_all(b"HTTP/1.1 302 Found\r\nLocation: https://api.openai.com/v1/responses\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "loopback request was not delivered directly"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(error) => panic!("{error}"),
            }
        }
    });
    let error = pool
        .get(format!("http://{address}/redirect"))
        .send()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("non-loopback"), "{error}");
    server.join().unwrap();
}

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
