//! Shared formatting for user-facing `codex resume` command hints.

use codex_protocol::ThreadId;
use codex_shell_command::parse_command::shlex_join;
use std::env;

pub const CODEX_CLI_NAME_ENV_VAR: &str = "CODEX_CLI_NAME";

pub fn resume_command(thread_name: Option<&str>, thread_id: Option<ThreadId>) -> Option<String> {
    let resume_target = thread_name
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| thread_id.map(|thread_id| thread_id.to_string()));
    resume_target.map(|target| {
        let command = cli_command_for_action("resume");
        let needs_double_dash = target.starts_with('-');
        let escaped = shlex_join(&[target]);
        if needs_double_dash {
            format!("{command} -- {escaped}")
        } else {
            format!("{command} {escaped}")
        }
    })
}

pub fn cli_command_for_action(action: &str) -> String {
    let cli_name = env::var(CODEX_CLI_NAME_ENV_VAR)
        .ok()
        .and_then(|name| {
            let name = name.trim();
            if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            }
        })
        .unwrap_or_else(|| "codex".to_string());

    shlex_join(&[cli_name, action.to_string()])
}

pub fn resume_hint(thread_name: Option<&str>, thread_id: Option<ThreadId>) -> Option<String> {
    let thread_id = thread_id?;
    match thread_name.filter(|name| !name.is_empty()) {
        Some(thread_name) => Some(format!(
            "codex resume, then select {thread_name} ({thread_id})"
        )),
        None => resume_command(/*thread_name*/ None, Some(thread_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serial_test::serial;
    use std::ffi::OsStr;

    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &OsStr) -> Self {
            let original = env::var_os(key);
            unsafe {
                env::set_var(key, value);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.original.take() {
                Some(value) => unsafe {
                    env::set_var(self.key, value);
                },
                None => unsafe {
                    env::remove_var(self.key);
                },
            }
        }
    }

    #[test]
    #[serial]
    fn prefers_name_over_id() {
        let thread_id = ThreadId::from_string("123e4567-e89b-12d3-a456-426614174000").unwrap();
        let command = resume_command(Some("my-thread"), Some(thread_id));
        assert_eq!(command, Some("codex resume my-thread".to_string()));
    }

    #[test]
    #[serial]
    fn formats_thread_id_when_name_is_missing() {
        let thread_id = ThreadId::from_string("123e4567-e89b-12d3-a456-426614174000").unwrap();
        let command = resume_command(/*thread_name*/ None, Some(thread_id));
        assert_eq!(
            command,
            Some("codex resume 123e4567-e89b-12d3-a456-426614174000".to_string())
        );
    }

    #[test]
    #[serial]
    fn returns_none_without_a_resume_target() {
        let command = resume_command(/*thread_name*/ None, /*thread_id*/ None);
        assert_eq!(command, None);
    }

    #[test]
    #[serial]
    fn quotes_thread_names_when_needed() {
        let command = resume_command(Some("-starts-with-dash"), /*thread_id*/ None);
        assert_eq!(
            command,
            Some("codex resume -- -starts-with-dash".to_string())
        );

        let command = resume_command(Some("two words"), /*thread_id*/ None);
        assert_eq!(command, Some("codex resume 'two words'".to_string()));

        let command = resume_command(Some("quote'case"), /*thread_id*/ None);
        assert_eq!(command, Some("codex resume \"quote'case\"".to_string()));
    }

    #[test]
    #[serial]
    fn resume_hint_names_picker_item_with_id() {
        let thread_id = ThreadId::from_string("123e4567-e89b-12d3-a456-426614174000").unwrap();
        let hint = resume_hint(Some("my-thread"), Some(thread_id));
        assert_eq!(
            hint,
            Some(
                "codex resume, then select my-thread (123e4567-e89b-12d3-a456-426614174000)"
                    .to_string()
            )
        );
    }

    #[test]
    #[serial]
    fn resume_hint_uses_direct_id_command_without_name() {
        let thread_id = ThreadId::from_string("123e4567-e89b-12d3-a456-426614174000").unwrap();
        let hint = resume_hint(/*thread_name*/ None, Some(thread_id));
        assert_eq!(
            hint,
            Some("codex resume 123e4567-e89b-12d3-a456-426614174000".to_string())
        );
    }

    #[test]
    #[serial]
    fn resume_hint_requires_thread_id() {
        let hint = resume_hint(Some("my-thread"), /*thread_id*/ None);
        assert_eq!(hint, None);
    }

    #[test]
    #[serial]
    fn resume_command_uses_cli_name_env_override() {
        let _guard = EnvVarGuard::set(CODEX_CLI_NAME_ENV_VAR, OsStr::new("marathon"));
        let thread_id = ThreadId::from_string("123e4567-e89b-12d3-a456-426614174000").unwrap();
        let command = resume_command(/*thread_name*/ None, Some(thread_id));
        assert_eq!(
            command,
            Some("marathon resume 123e4567-e89b-12d3-a456-426614174000".to_string())
        );
    }

    #[test]
    #[serial]
    fn cli_command_for_action_quotes_cli_name_env_override() {
        let _guard = EnvVarGuard::set(
            CODEX_CLI_NAME_ENV_VAR,
            OsStr::new("/tmp/My Marathon/bin/marathon"),
        );
        assert_eq!(
            cli_command_for_action("resume"),
            "'/tmp/My Marathon/bin/marathon' resume"
        );
    }
}
