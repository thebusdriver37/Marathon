use super::enforce;
use crate::config::Config;
use crate::config::ConfigOverrides;
use codex_config::config_toml::ConfigToml;
use codex_config::types::OtelExporterKind;
use codex_features::Feature;
use codex_utils_absolute_path::AbsolutePathBuf;
use pretty_assertions::assert_eq;

#[tokio::test]
async fn local_runtime_overrides_cloud_settings_without_removing_explicit_tools() {
    let home = tempfile::tempdir().unwrap();
    let cfg: ConfigToml = toml::from_str(
        r#"
model_provider = "marathon-local"
check_for_update_on_startup = true
analytics.enabled = true
feedback.enabled = true
otel.metrics_exporter = "statsig"
features.apps = true
features.plugins = true
features.hooks = true
[model_providers.marathon-local]
name = "Marathon Local"
base_url = "http://127.0.0.1:18111/v1"
wire_api = "responses"
requires_openai_auth = false
[mcp_servers.explicit-local-tool]
command = "local-tool"
"#,
    )
    .unwrap();
    let mut config = Config::load_from_base_config_with_overrides(
        cfg,
        ConfigOverrides::default(),
        AbsolutePathBuf::from_absolute_path(home.path()).unwrap(),
    )
    .await
    .unwrap();
    enforce(&mut config).unwrap();
    assert_eq!(
        (
            config.check_for_update_on_startup,
            config.analytics_enabled,
            config.feedback_enabled
        ),
        (false, Some(false), false),
    );
    assert_eq!(
        (
            config.otel.exporter,
            config.otel.trace_exporter,
            config.otel.metrics_exporter
        ),
        (
            OtelExporterKind::None,
            OtelExporterKind::None,
            OtelExporterKind::None
        ),
    );
    assert!(!config.features.enabled(Feature::Apps));
    assert!(!config.features.enabled(Feature::Plugins));
    assert!(!config.features.enabled(Feature::CodexHooks));
    assert!(config.mcp_servers.get().contains_key("explicit-local-tool"));
}

#[tokio::test]
async fn local_runtime_rejects_cloud_model_provider() {
    let home = tempfile::tempdir().unwrap();
    let mut config = Config::load_from_base_config_with_overrides(
        ConfigToml::default(),
        ConfigOverrides::default(),
        AbsolutePathBuf::from_absolute_path(home.path()).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        enforce(&mut config).unwrap_err().kind(),
        std::io::ErrorKind::PermissionDenied
    );
}
