//! Secret scan for corpus inputs.
//!
//! The scan only reports. A file with a hit is dropped by the builder and the
//! hit is counted; the builder never edits text.

use regex::Regex;
use std::sync::OnceLock;

struct Pattern {
    name: &'static str,
    regex: Regex,
}

fn patterns() -> &'static [Pattern] {
    static PATTERNS: OnceLock<Vec<Pattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let specs: &[(&'static str, &'static str)] = &[
            ("aws_access_key", r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}"),
            (
                "generic_api_key",
                r#"(?i)(?:api[_-]?key|apikey|access[_-]?token|secret[_-]?key)\s*[:=]\s*['"]?[A-Za-z0-9_\-]{20,}"#,
            ),
            (
                "bearer_token",
                r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}=*",
            ),
            (
                "private_key",
                r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            ),
            ("github_token", r"gh[pousr]_[A-Za-z0-9]{36}"),
            ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
            (
                "jwt",
                r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
            ),
            ("google_api_key", r"AIza[0-9A-Za-z\-_]{35}"),
            ("stripe_live_key", r"sk_live_[0-9a-zA-Z]{24,}"),
            (
                "url_with_credentials",
                r"(?i)https?://[^:\s]+:[^@\s]+@",
            ),
        ];
        specs
            .iter()
            .map(|(name, pattern)| Pattern {
                name,
                regex: Regex::new(pattern).expect("corpus secret pattern compiles"),
            })
            .collect()
    })
}

/// Names of all secret recognizers, sorted for a stable manifest.
pub fn pattern_names() -> Vec<String> {
    let mut names: Vec<String> = patterns().iter().map(|p| p.name.to_string()).collect();
    names.sort();
    names
}

/// Scan text and return one recognizer name per match. Order is stable.
pub fn scan(text: &str) -> Vec<&'static str> {
    let mut hits = Vec::new();
    for pattern in patterns() {
        for _ in pattern.regex.find_iter(text) {
            hits.push(pattern.name);
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_aws_key() {
        let hits = scan("token = \"AKIAIOSFODNN7EXAMPLE\"");
        assert!(hits.contains(&"aws_access_key"));
    }

    #[test]
    fn clean_text_has_no_hits() {
        assert!(scan("fn main() { println!(\"hello\"); }").is_empty());
    }
}
