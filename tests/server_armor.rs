//! End-to-end protection of the CLI server's request boundary (no model needed).
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn request(armor: bool, endpoint: &str, stream: bool, sensitive: bool) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let mut command = Command::new(env!("CARGO_BIN_EXE_ferrisres"));
    command.args(["serve", "--host", "127.0.0.1", "--port", &port.to_string()]);
    if armor { command.arg("--armor"); }
    let mut server = Server(command.stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut socket = loop {
        if let Ok(socket) = TcpStream::connect(("127.0.0.1", port)) { break socket; }
        assert!(server.0.try_wait().unwrap().is_none(), "server exited before listening");
        assert!(Instant::now() < deadline, "server did not start in 15s");
        std::thread::sleep(Duration::from_millis(20));
    };
    socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    socket.set_write_timeout(Some(Duration::from_secs(10))).unwrap();
    let content = if sensitive { "Email me at alice@example.com" } else { "Hello" };
    // Keep compact role-first objects compatible with the existing request parser.
    let body = if endpoint == "/v1/chat/completions" {
        format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"stream":{stream}}}"#)
    } else {
        format!(r#"{{"prompt":"{content}","stream":{stream}}}"#)
    };
    write!(socket, "POST {endpoint} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    response
}

#[test]
fn armor_blocks_sensitive_inputs_on_both_completion_endpoints() {
    for endpoint in ["/v1/chat/completions", "/v1/completions"] {
        for stream in [false, true] {
            let response = request(true, endpoint, stream, true);
            assert!(response.starts_with("HTTP/1.1 403"), "{endpoint} stream={stream}: {response}");
            assert!(!response.contains("alice@example.com"), "blocked input must not be echoed");
        }
    }
}

#[test]
fn armor_allows_benign_input() {
    for endpoint in ["/v1/chat/completions", "/v1/completions"] {
        assert!(request(true, endpoint, false, false).starts_with("HTTP/1.1 200"));
    }
}

#[test]
fn armor_is_opt_in() {
    for endpoint in ["/v1/chat/completions", "/v1/completions"] {
        assert!(request(false, endpoint, false, true).starts_with("HTTP/1.1 200"));
    }
}
