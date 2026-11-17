//! Fail-closed overrides for Marathon, independent of user/project config precedence.

use super::Config;
use codex_config::types::AuthCredentialsStoreMode;
use codex_config::types::OtelExporterKind;
use codex_features::Feature;

pub(super) fn enforce(config: &mut Config) -> std::io::Result<()> {
    if config.model_provider.requires_openai_auth {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Marathon requires its local model provider",
        ));
    }
    codex_http_client::validate_local_runtime_url(
        config.model_provider.base_url.as_deref().unwrap_or(""),
    )?;
    config.check_for_update_on_startup = false;
    config.analytics_enabled = Some(false);
    config.feedback_enabled = false;
    config.otel.exporter = OtelExporterKind::None;
    config.otel.trace_exporter = OtelExporterKind::None;
    config.otel.metrics_exporter = OtelExporterKind::None;
    config.otel.log_user_prompt = false;
    config.cli_auth_credentials_store_mode = AuthCredentialsStoreMode::File;
    config.orchestrator_skills_enabled = false;
    config.orchestrator_mcp_enabled = false;
    config.show_tooltips = false;
    config.tui_pet = None;
    config.notify = None;
    for feature in [
        Feature::Apps,
        Feature::Plugins,
        Feature::RemotePlugin,
        Feature::ToolSuggest,
        Feature::RecommendedPlugins,
        Feature::CodexHooks,
        Feature::PluginHooks,
    ] {
        config.features.disable(feature)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "local_runtime_tests.rs"]
mod tests;
