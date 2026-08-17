"""Arrow-key terminal workflow for selecting and running local models."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .catalog import (
    Model,
    Profile,
    ROOT_DIR,
    backend_for,
    backends,
    discover_models,
    find_model,
    find_profile,
    format_size,
    profiles_for_model,
    settings,
)
from .dyno import OBJECTIVES, candidate_profiles, run_tuning
from .frontends import _codex_binary, direct_chat, run_codex, run_hermes
from .model_library import (
    RECOMMENDED_QWEN_REPOSITORY,
    download_huggingface_gguf,
    list_huggingface_ggufs,
    register_model_root,
)
from .remote import (
    RemoteRuntime,
    fetch_remote_catalog,
    load_remote_selection,
    save_remote_selection,
)
from .runtime import Runtime, load_selection, save_selection


FRONTEND_NAMES = {
    "codex": "Codex",
    "hermes": "Hermes Agent",
    "direct": "Direct Chat",
}


def _dyno_supported(model: Model) -> bool:
    backend = backends().get(model.family.backend)
    return backend is not None and backend.kind == "llama_cpp"


@dataclass
class Selection:
    model: Model
    profile: Profile
    frontend: str


@dataclass(frozen=True)
class MenuItem:
    label: str
    description: str
    value: str
    badge: str = ""


MENU_STYLE = Style.from_dict(
    {
        "logo": "bold #ff5fff",
        "tagline": "#888888",
        "context": "#dddddd",
        "context.detail": "#5fd7ff",
        "title": "bold #5fd7ff",
        "subtitle": "#888888",
        "item": "#eeeeee",
        "item.badge": "bold #5fd7ff",
        "item.description": "#888888",
        "selected": "bold #000000 bg:#5fd7ff",
        "selected.description": "#000000 bg:#5fd7ff",
        "help": "#888888",
    }
)


def _menu_content(
    title: str,
    subtitle: str,
    context: tuple[str, ...],
    items: list[MenuItem],
    selected: int,
    allow_back: bool,
) -> StyleAndTextTuples:
    content: StyleAndTextTuples = [
        ("class:logo", "  MARATHON"),
        ("class:tagline", "   local AI, ready when you are\n"),
        ("class:tagline", "  ────────────────────────────────────────\n\n"),
    ]
    if context:
        content.append(("class:context", f"  {context[0]}\n"))
        for line in context[1:]:
            content.append(("class:context.detail", f"  {line}\n"))
        content.append(("", "\n"))
    content.extend(
        [
            ("class:title", f"  {title}\n"),
            ("class:subtitle", f"  {subtitle}\n\n"),
        ]
    )
    for index, item in enumerate(items):
        active = index == selected
        item_style = "class:selected" if active else "class:item"
        description_style = (
            "class:selected.description" if active else "class:item.description"
        )
        marker = "❯" if active else " "
        badge = f"  {item.badge}" if item.badge else ""
        content.append((item_style, f"  {marker} {item.label}{badge}  \n"))
        content.append((description_style, f"      {item.description}  \n"))
    back_help = "   Esc/q back" if allow_back else ""
    content.append(("class:help", f"  ↑/↓ move   Enter select{back_help}\n"))
    return content


def _arrow_menu(
    console: Console,
    title: str,
    subtitle: str,
    items: list[MenuItem],
    selected: int = 0,
    *,
    allow_back: bool = True,
    context: tuple[str, ...] = (),
) -> int | None:
    if not items:
        return None
    if not sys.stdin.isatty():
        raise RuntimeError("Marathon's interactive menu requires a terminal")
    state = {"selected": min(max(selected, 0), len(items) - 1), "result": None}
    bindings = KeyBindings()

    def move(offset: int) -> None:
        state["selected"] = (int(state["selected"]) + offset) % len(items)

    @bindings.add("up")
    @bindings.add("k")
    def _up(event: KeyPressEvent) -> None:
        move(-1)

    @bindings.add("down")
    @bindings.add("j")
    def _down(event: KeyPressEvent) -> None:
        move(1)

    @bindings.add("enter")
    def _select(event: KeyPressEvent) -> None:
        state["result"] = int(state["selected"])
        event.app.exit()

    @bindings.add("c-c")
    def _interrupt(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt())

    if allow_back:
        @bindings.add("escape")
        @bindings.add("q")
        def _back(event: KeyPressEvent) -> None:
            event.app.exit()

    control = FormattedTextControl(
        lambda: _menu_content(
            title,
            subtitle,
            context,
            items,
            int(state["selected"]),
            allow_back,
        ),
        focusable=True,
    )
    application: Application[None] = Application(
        layout=Layout(Window(control, wrap_lines=True, always_hide_cursor=True)),
        key_bindings=bindings,
        full_screen=True,
        mouse_support=False,
        style=MENU_STYLE,
    )
    console.clear()
    application.run()
    result = state["result"]
    return int(result) if result is not None else None


def _model_items(models: list[Model]) -> list[MenuItem]:
    return [
        MenuItem(
            model.display_name,
            f"Central GGUF · {len(profiles_for_model(model))} runtime profiles",
            model.id,
            format_size(model.size_bytes),
        )
        for model in models
    ]


def _profile_items(model: Model) -> list[MenuItem]:
    return [
        MenuItem(
            profile.display_name,
            profile.description,
            profile.id,
            f"{profile.context // 1024}K · {profile.confidence}",
        )
        for profile in profiles_for_model(model)
    ]


def _choose_installed_model(console: Console, models: list[Model]) -> Selection | None:
    chosen = _arrow_menu(
        console,
        "Choose your model",
        "Marathon will choose the safest compatible runtime profile.",
        _model_items(models),
        0,
        context=("Installed GGUF models", f"{len(models)} found"),
    )
    if chosen is None:
        return None
    model = models[chosen]
    try:
        profile = find_profile(model, None, "codex")
    except ValueError:
        compatible = [
            profile for profile in profiles_for_model(model) if profile.supports("codex")
        ]
        if not compatible:
            raise ValueError(f"{model.display_name} has no Codex-compatible profile")
        profile = compatible[0]
    return Selection(model, profile, "codex")


def _download_gguf(console: Console, repository: str) -> Path | None:
    try:
        with console.status(
            f"[bold magenta]Reading {repository} from Hugging Face...[/bold magenta]",
            spinner="dots",
        ):
            files = list_huggingface_ggufs(repository)
    except Exception as error:
        console.print(Panel(str(error), title="Cannot read repository", border_style="red"))
        Prompt.ask("Press Enter to continue", default="")
        return None
    if not files:
        console.print(Panel("No single-file GGUF models were found.", border_style="yellow"))
        Prompt.ask("Press Enter to continue", default="")
        return None

    common_quants = ("Q4_K_M", "UD-Q4_K_XL", "Q5_K_M", "Q6_K", "Q8_0")
    primary = [
        next((item for item in files if item.quant == quant), None)
        for quant in common_quants
    ]
    primary = [item for item in primary if item is not None]
    remaining = [item for item in files if item not in primary]
    items = [
        MenuItem(
            item.quant,
            item.filename,
            str(index),
            (
                f"{format_size(item.size_bytes)} · recommended"
                if item.size_bytes is not None and item.quant == "Q4_K_M"
                else format_size(item.size_bytes)
                if item.size_bytes is not None
                else "size unknown"
            ),
        )
        for item in primary
    ]
    if remaining:
        items.append(
            MenuItem(
                "Another quant",
                "Enter an exact quant or GGUF filename from this repository.",
                "other",
                f"{len(remaining)} more",
            )
        )
    chosen = _arrow_menu(
        console,
        "Choose a quant",
        "Q4_K_M is the best starting point for most systems.",
        items,
        0,
        context=(repository, "The exact repository revision will be recorded"),
    )
    if chosen is None:
        return None
    if chosen < len(primary):
        selected = primary[chosen]
    else:
        console.clear()
        table = Table(title=f"Other GGUF files in {repository}")
        table.add_column("Quant", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Filename", style="dim")
        for item in remaining:
            table.add_row(
                item.quant,
                format_size(item.size_bytes) if item.size_bytes is not None else "unknown",
                item.filename,
            )
        console.print(table)
        value = Prompt.ask("Exact quant or GGUF filename").strip().lower()
        matches = [
            item
            for item in remaining
            if value in {item.quant.lower(), item.filename.lower()}
        ]
        if len(matches) != 1:
            console.print("[bold red]Enter one exact quant or filename from the table.[/bold red]")
            Prompt.ask("Press Enter to continue", default="")
            return None
        selected = matches[0]
    console.clear()
    console.print(
        f"[bold magenta]Downloading {selected.filename}[/bold magenta]\n"
        f"[dim]{selected.repository} at {selected.revision[:12]}[/dim]\n"
    )
    try:
        downloaded = download_huggingface_gguf(selected, settings().model_root)
    except Exception as error:
        console.print(Panel(str(error), title="Download failed", border_style="red"))
        Prompt.ask("Press Enter to continue", default="")
        return None
    console.print(f"\n[green]Downloaded:[/green] {downloaded}")
    return downloaded


def _setup_model_selection(console: Console) -> Selection | None:
    while True:
        models = discover_models()
        items: list[MenuItem] = []
        if models:
            items.append(
                MenuItem(
                    "Use an installed model",
                    "Choose from compatible GGUF files Marathon already found.",
                    "installed",
                    f"{len(models)} found",
                )
            )
        items.extend(
            [
                MenuItem(
                    "Download Qwen 3.8 27B",
                    "Choose a quant from the curated Unsloth GGUF repository.",
                    "recommended",
                    "recommended",
                ),
                MenuItem(
                    "Add an existing model folder",
                    "Use GGUF files in place without copying or moving them.",
                    "folder",
                ),
                MenuItem(
                    "Use another Hugging Face repository",
                    "Enter an owner/name repository and choose its GGUF file.",
                    "repository",
                    "advanced",
                ),
                MenuItem("Exit setup", "Leave the system unchanged.", "quit"),
            ]
        )
        chosen = _arrow_menu(
            console,
            "Set up your model",
            "Use what you already have or let Marathon download one.",
            items,
            0,
            allow_back=False,
            context=("First-run setup", str(settings().model_root)),
        )
        assert chosen is not None
        action = items[chosen].value
        if action == "quit":
            return None
        if action == "installed":
            selection = _choose_installed_model(console, models)
            if selection is not None:
                return selection
            continue
        if action == "folder":
            console.clear()
            value = Prompt.ask("Folder containing GGUF models").strip()
            if not value:
                continue
            candidate = Path(value).expanduser()
            if candidate.is_file() and candidate.suffix.lower() == ".gguf":
                candidate = candidate.parent
            try:
                root = register_model_root(candidate)
            except ValueError as error:
                console.print(f"[bold red]{error}[/bold red]")
                Prompt.ask("Press Enter to continue", default="")
                continue
            found = discover_models(root)
            if not found:
                console.print(f"[yellow]No GGUF models found under {root}.[/yellow]")
                Prompt.ask("Press Enter to continue", default="")
                continue
            selection = _choose_installed_model(console, discover_models())
            if selection is not None:
                return selection
            continue
        repository = RECOMMENDED_QWEN_REPOSITORY
        if action == "repository":
            console.clear()
            repository = Prompt.ask("Hugging Face repository (owner/name)").strip()
            if not repository:
                continue
        downloaded = _download_gguf(console, repository)
        if downloaded is None:
            continue
        refreshed = discover_models()
        selected = next(
            (model for model in refreshed if model.path.resolve() == downloaded.resolve()),
            None,
        )
        if selected is None:
            console.print("[bold red]The downloaded GGUF could not be indexed.[/bold red]")
            Prompt.ask("Press Enter to continue", default="")
            continue
        return Selection(selected, find_profile(selected, None, "codex"), "codex")


def _confirm_install(console: Console, component: str, description: str) -> bool:
    choice = _arrow_menu(
        console,
        f"Install {component}",
        "This is a one-time setup step.",
        [
            MenuItem(f"Install {component}", description, "install", "recommended"),
            MenuItem("Exit", "Leave the current installation unchanged.", "quit"),
        ],
        0,
        allow_back=False,
        context=("Marathon setup", component),
    )
    return choice == 0


def _run_install_command(console: Console, label: str, command: list[str]) -> bool:
    console.clear()
    console.print(f"[bold magenta]{label}[/bold magenta]\n")
    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    if result.returncode == 0:
        return True
    console.print(
        Panel(
            f"The installer exited with status {result.returncode}.",
            title=f"{label} did not finish",
            border_style="red",
        )
    )
    Prompt.ask("Press Enter to continue", default="")
    return False


def _ensure_local_tools(
    console: Console, selection: Selection, frontend: str = "codex"
) -> bool:
    try:
        backend_for(selection.model, selection.profile)
    except ValueError as error:
        backend_id = selection.profile.backend or selection.model.family.backend
        if backend_id != "upstream":
            console.print(Panel(str(error), title="Backend unavailable", border_style="red"))
            return False
        if not _confirm_install(
            console,
            "llama.cpp",
            "Build the pinned local inference engine for this machine.",
        ):
            return False
        if not _run_install_command(
            console,
            "Building llama.cpp",
            [str(ROOT_DIR / "bin" / "marathon"), "setup-llama"],
        ):
            return False
        try:
            backend_for(selection.model, selection.profile)
        except ValueError as install_error:
            console.print(
                Panel(str(install_error), title="Backend unavailable", border_style="red")
            )
            return False

    if frontend != "codex":
        return True

    codex = _codex_binary()
    if Path(codex).is_file() or shutil.which(codex):
        return True
    if not _confirm_install(
        console,
        "Marathon Codex",
        "Build the pinned terminal agent with Marathon's local-model patches.",
    ):
        return False
    return _run_install_command(
        console,
        "Building Marathon Codex",
        [str(ROOT_DIR / "bin" / "marathon"), "build-codex"],
    )


def _choose_model_profile(
    console: Console,
    models: list[Model],
    selection: Selection,
    *,
    library_context: tuple[str, str] | None = None,
) -> Selection | None:
    model_index = next(
        (index for index, model in enumerate(models) if model.id == selection.model.id), 0
    )
    while True:
        chosen_model = _arrow_menu(
            console,
            "Choose a model",
            "Enter drills into that model's runtime profiles.",
            _model_items(models),
            model_index,
            context=library_context or ("Central model library", str(settings().model_root)),
        )
        if chosen_model is None:
            return None
        model = models[chosen_model]
        model_index = chosen_model
        profiles = list(profiles_for_model(model))
        profile_index = next(
            (
                index
                for index, profile in enumerate(profiles)
                if model.id == selection.model.id and profile.id == selection.profile.id
            ),
            next(
                (index for index, profile in enumerate(profiles) if profile.id == model.family.default_profile),
                0,
            ),
        )
        chosen_profile = _arrow_menu(
            console,
            model.display_name,
            "Choose how this model should run. Esc returns to the model list.",
            _profile_items(model),
            profile_index,
            context=("Selected model", f"{model.quant} · {format_size(model.size_bytes)}"),
        )
        if chosen_profile is None:
            continue
        profile = profiles[chosen_profile]
        frontend = selection.frontend if profile.supports(selection.frontend) else profile.frontends[0]
        return Selection(model, profile, frontend)


def _confirm_experimental(console: Console, profile: Profile) -> bool:
    if profile.confidence != "experimental":
        return True
    choice = _arrow_menu(
        console,
        "Experimental profile",
        f"{profile.display_name} has not completed stability validation on this machine.",
        [
            MenuItem("Go back", "Keep the current setup unchanged.", "back"),
            MenuItem("Start anyway", "Run the experimental configuration.", "start", "experimental"),
        ],
        0,
        context=(profile.display_name, f"{profile.context:,} token context"),
    )
    return choice == 1


def _dyno_items() -> list[MenuItem]:
    badges = {
        "balanced": "recommended",
        "speed": "latency",
        "context": "context",
        "quality": "precision",
        "efficiency": "power",
    }
    return [
        MenuItem(label, description, objective, badges[objective])
        for objective, (label, description) in OBJECTIVES.items()
    ]


def _show_dyno_results(console: Console, summary) -> None:
    console.clear()
    winner = summary.winner
    console.print(
        Panel(
            f"[bold green]{winner.candidate.label}[/bold green]\n"
            f"{winner.prompt_tps:.1f} prompt tok/s · {winner.decode_tps:.1f} decode tok/s · "
            f"{winner.loaded_context:,} tokens\n\n"
            f"Saved as [cyan]Dyno · {OBJECTIVES[summary.objective][0]}[/cyan]",
            title="Dyno found a winner",
            border_style="green",
        )
    )
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Trial")
    table.add_column("Prompt", justify="right")
    table.add_column("Decode", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("Result")
    for result in summary.results:
        table.add_row(
            result.candidate.label,
            f"{result.prompt_tps:.1f}" if result.success else "—",
            f"{result.decode_tps:.1f}" if result.success else "—",
            f"{result.loaded_context // 1024}K" if result.loaded_context else "—",
            f"{result.average_power_w:.0f}W" if result.average_power_w else "—",
            "[green]pass[/green]" if result.success else f"[red]{result.error[:38]}[/red]",
        )
    console.print(table)
    console.print(f"[dim]Full results: {summary.result_dir}[/dim]")
    Prompt.ask("Press Enter to continue", default="")


def _run_dyno_flow(console: Console, selection: Selection) -> Selection | None:
    # Tune from the shipped family default, not from a previous local winner or
    # a deliberately constrained quick-chat profile.
    baseline = find_profile(selection.model, selection.model.family.default_profile)
    chosen = _arrow_menu(
        console,
        "What should Dyno optimize?",
        "Choose one priority. Every trial must pass the same safety gates.",
        _dyno_items(),
        0,
        context=(selection.model.display_name, f"Baseline · {baseline.display_name}"),
    )
    if chosen is None:
        return None
    objective = list(OBJECTIVES)[chosen]
    candidates = candidate_profiles(selection.model, baseline, objective)
    confirmation = _arrow_menu(
        console,
        f"Tune for {OBJECTIVES[objective][0]}",
        "Candidates run in the foreground; GPUs are freed between trials.",
        [
            MenuItem("Start tuning", f"Run {len(candidates)} deterministic trials and save the winner.", "start", f"{len(candidates)} trials"),
            MenuItem("Go back", "Do not change the current setup.", "back"),
        ],
        0,
        context=(selection.model.display_name, "Known-good profiles are never overwritten"),
    )
    if confirmation != 0:
        return None
    with console.status("[bold magenta]Dyno is preparing the first trial…[/bold magenta]", spinner="dots") as status:
        summary = run_tuning(
            selection.model,
            baseline,
            objective,
            lambda message: status.update(f"[magenta]{message}[/magenta]"),
        )
    _show_dyno_results(console, summary)
    tuned = find_profile(selection.model, f"dyno-{objective}")
    frontend = selection.frontend if tuned.supports(selection.frontend) else tuned.frontends[0]
    return Selection(selection.model, tuned, frontend)


def _home_items(
    selection: Selection, *, warm: bool, allow_tune: bool = True
) -> list[MenuItem]:
    suffix = "Model is already loaded." if warm else "Load the model and open Codex."
    items: list[MenuItem] = []
    if selection.profile.supports("codex"):
        items.append(MenuItem("Start Codex" if not warm else "Open Codex", suffix, "codex", "default"))
    if selection.profile.supports("hermes"):
        items.append(
            MenuItem(
                "Start Hermes" if not warm else "Open Hermes",
                "Use the normal Hermes tools, memory, skills, and project rules.",
                "hermes",
                "agent",
            )
        )
    items.append(
        MenuItem(
            "Start Direct Chat" if not warm else "Open Direct Chat",
            "Talk directly to the model without an agent harness.",
            "direct",
        )
    )
    items.append(
        MenuItem(
            "Choose model / profile",
            "Choose a model, then one of its profiles.",
            "change",
        )
    )
    if not warm and allow_tune and _dyno_supported(selection.model):
        items.append(
            MenuItem(
                "Tune / benchmark",
                "Let Dyno find a machine-specific profile for this model.",
                "tune",
                "advanced",
            )
        )
    items.append(
        MenuItem(
            "Quit" if not warm else "Stop backend and quit",
            "Free all GPU memory and exit Marathon.",
            "quit",
        )
    )
    return items


def _home(
    console: Console,
    models: list[Model],
    selection: Selection,
    *,
    warm: bool,
    preferred_frontend: str | None = None,
    allow_tune: bool = True,
    location: str | None = None,
    library_context: tuple[str, str] | None = None,
) -> tuple[str, Selection]:
    while True:
        items = _home_items(selection, warm=warm, allow_tune=allow_tune)
        selected = next(
            (
                index
                for index, item in enumerate(items)
                if item.value == (preferred_frontend or selection.frontend)
            ),
            0,
        )
        chosen = _arrow_menu(
            console,
            "What would you like to do?",
            "Your last setup is remembered. Enter starts it.",
            items,
            selected,
            allow_back=False,
            context=(
                ("● Backend ready" if warm else "Current setup"),
                *((location,) if location else ()),
                selection.model.display_name,
                f"{selection.profile.display_name} · {selection.profile.context:,} tokens · "
                f"{FRONTEND_NAMES[selection.frontend]}",
            ),
        )
        assert chosen is not None
        action = items[chosen].value
        preferred_frontend = None
        if action == "change":
            previous = (selection.model.id, selection.profile.id)
            changed = _choose_model_profile(
                console,
                models,
                selection,
                library_context=library_context,
            )
            if changed:
                selection = changed
                current = (selection.model.id, selection.profile.id)
                if warm and current != previous:
                    return "change", selection
            continue
        if action in {"codex", "hermes", "direct"}:
            selection.frontend = action
            if not _confirm_experimental(console, selection.profile):
                continue
        return action, selection


def _initial_selection(
    models: list[Model], remembered: dict[str, str] | None = None
) -> Selection:
    remembered = load_selection() if remembered is None else remembered
    try:
        model = find_model(remembered.get("model", ""), models)
    except ValueError:
        preferred = [item for item in models if item.family.id == "qwen3.8-27b"]
        model = preferred[0] if preferred else models[0]
    frontend = remembered.get("frontend", "codex")
    try:
        profile = find_profile(model, remembered.get("profile"), frontend)
    except ValueError:
        frontend = "codex"
        profile = find_profile(model, None, frontend)
    return Selection(model, profile, frontend)


def _launch_frontend(
    console: Console,
    runtime: Runtime | RemoteRuntime,
    frontend: str,
    extra_args: list[str] | None = None,
) -> None:
    console.clear()
    if frontend == "direct":
        direct_chat(runtime, console)
        return
    code = (
        run_hermes(runtime, extra_args)
        if frontend == "hermes"
        else run_codex(runtime, extra_args)
    )
    if code not in (0, 130):
        console.print(
            f"[yellow]{FRONTEND_NAMES[frontend]} exited with status {code}.[/yellow]"
        )


def _apply_initial_frontend(
    selection: Selection, initial_frontend: str | None
) -> Selection:
    if initial_frontend:
        try:
            selection.profile = find_profile(selection.model, selection.profile.id, initial_frontend)
        except ValueError:
            selection.profile = find_profile(selection.model, None, initial_frontend)
        selection.frontend = initial_frontend
    return selection


def _run_runtime_dashboard(
    console: Console,
    models: list[Model],
    selection: Selection,
    *,
    initial_frontend: str | None,
    runtime_factory: Callable[[Selection], Runtime | RemoteRuntime],
    remember: Callable[[Model, Profile, str], None],
    allow_tune: bool,
    location: str | None,
    library_context: tuple[str, str] | None,
    preparing_message: str,
    stopping_message: str,
    stopped_message: str,
    error_title: str,
    ensure_local_tools: bool,
) -> int:
    preferred = initial_frontend
    while True:
        try:
            action, selection = _home(
                console,
                models,
                selection,
                warm=False,
                preferred_frontend=preferred,
                allow_tune=allow_tune,
                location=location,
                library_context=library_context,
            )
        except KeyboardInterrupt:
            console.print()
            return 130
        preferred = None
        if action == "quit":
            return 0
        if action == "tune":
            try:
                tuned = _run_dyno_flow(console, selection)
                if tuned:
                    selection = tuned
                    remember(selection.model, selection.profile, selection.frontend)
            except KeyboardInterrupt:
                console.print("\n[yellow]Dyno stopped. GPUs are being freed.[/yellow]")
            except Exception as error:
                console.print(Panel(str(error), title="Dyno could not finish", border_style="red"))
                Prompt.ask("Press Enter to return", default="")
            continue
        if ensure_local_tools and not _ensure_local_tools(console, selection, action):
            continue
        remember(selection.model, selection.profile, action)
        runtime = runtime_factory(selection)
        try:
            with console.status(preparing_message, spinner="dots") as status:
                runtime.start(lambda message: status.update(f"[magenta]{message}[/magenta]"))
            _launch_frontend(console, runtime, action)
            while True:
                warm_action, selection = _home(
                    console,
                    models,
                    selection,
                    warm=True,
                    allow_tune=allow_tune,
                    location=location,
                    library_context=library_context,
                )
                if warm_action == "quit":
                    action = "quit"
                    break
                if warm_action == "change":
                    action = "switch"
                    break
                remember(selection.model, selection.profile, warm_action)
                _launch_frontend(console, runtime, warm_action)
        except KeyboardInterrupt:
            runtime.record("runtime.interrupted", {}, level="error")
            action = "quit"
        except Exception as error:
            runtime.record("runtime.error", {"error": str(error)}, level="error")
            console.print(Panel(str(error), title=error_title, border_style="red"))
            Prompt.ask("Press Enter to return", default="")
            action = "switch"
        finally:
            with console.status(stopping_message, spinner="dots"):
                runtime.cleanup()
        if action == "quit":
            console.print(stopped_message)
            return 0


def run_dashboard(initial_frontend: str | None = None) -> int:
    console = Console()
    models = discover_models()
    if not models:
        selection = _setup_model_selection(console)
        if selection is None:
            return 0
        selection = _apply_initial_frontend(selection, initial_frontend)
        models = discover_models()
    else:
        selection = _apply_initial_frontend(
            _initial_selection(models), initial_frontend
        )
    return _run_runtime_dashboard(
        console,
        models,
        selection,
        initial_frontend=initial_frontend,
        runtime_factory=lambda current: Runtime(current.model, current.profile),
        remember=save_selection,
        allow_tune=True,
        location=None,
        library_context=None,
        preparing_message="[bold magenta]Preparing GPUs…[/bold magenta]",
        stopping_message="[yellow]Stopping backend and freeing GPUs…[/yellow]",
        stopped_message="[green]Backend stopped. GPUs are free.[/green]",
        error_title="Marathon could not start",
        ensure_local_tools=True,
    )


def run_codex_default(codex_args: list[str] | None = None) -> int:
    """Start the remembered model and Codex without an intermediate menu."""

    console = Console()
    models = discover_models()
    if models:
        selection = _apply_initial_frontend(_initial_selection(models), "codex")
    else:
        selection = _setup_model_selection(console)
        if selection is None:
            return 0
    if not _ensure_local_tools(console, selection):
        return 2
    save_selection(selection.model, selection.profile, "codex")
    runtime = Runtime(selection.model, selection.profile)
    result = 0
    try:
        with console.status("[bold magenta]Preparing local Codex...[/bold magenta]", spinner="dots") as status:
            runtime.start(lambda message: status.update(f"[magenta]{message}[/magenta]"))
        _launch_frontend(console, runtime, "codex", codex_args)
    except KeyboardInterrupt:
        runtime.record("runtime.interrupted", {}, level="error")
        result = 130
    except Exception as error:
        runtime.record("runtime.error", {"error": str(error)}, level="error")
        console.print(Panel(str(error), title="Marathon could not start", border_style="red"))
        result = 2
    finally:
        with console.status("[yellow]Stopping backend and freeing GPUs...[/yellow]", spinner="dots"):
            runtime.cleanup()
    return result


def run_setup_dashboard() -> int:
    """Configure the local model library without starting a backend."""

    console = Console()
    selection = _setup_model_selection(console)
    if selection is None:
        return 0
    if not _ensure_local_tools(console, selection):
        return 2
    save_selection(selection.model, selection.profile, "codex")
    console.clear()
    console.print(
        Panel.fit(
            f"[bold green]Marathon is ready[/bold green]\n"
            f"{selection.model.display_name}\n"
            f"{selection.profile.display_name} · {selection.profile.context:,} requested tokens\n\n"
            "Run [bold]marathon[/bold] from the project you want Codex to edit.",
            border_style="green",
        )
    )
    return 0


def run_remote_dashboard(host: str, initial_frontend: str | None = None) -> int:
    """Run Codex on this machine against a foreground GPU host over SSH."""

    console = Console()
    try:
        with console.status(
            f"[bold magenta]Reading models from {host}…[/bold magenta]",
            spinner="dots",
        ):
            remote = fetch_remote_catalog(host)
    except Exception as error:
        console.print(Panel(str(error), title="Remote Marathon unavailable", border_style="red"))
        return 2
    models = remote.models
    if not models:
        console.print(f"[bold red]No GGUF models found on {host}.[/bold red]")
        return 2
    selection = _apply_initial_frontend(
        _initial_selection(models, load_remote_selection(host)), initial_frontend
    )
    return _run_runtime_dashboard(
        console,
        models,
        selection,
        initial_frontend=initial_frontend,
        runtime_factory=lambda current: RemoteRuntime(
            host, remote.router_port, current.model, current.profile
        ),
        remember=lambda model, profile, frontend: save_remote_selection(
            host, model, profile, frontend
        ),
        allow_tune=False,
        location=f"Remote GPU host · {host}",
        library_context=("Remote model library", host),
        preparing_message=(
            f"[bold magenta]Connecting to {host} and preparing GPUs…[/bold magenta]"
        ),
        stopping_message=(
            f"[yellow]Stopping backend on {host} and freeing GPUs…[/yellow]"
        ),
        stopped_message=f"[green]Backend on {host} stopped. GPUs are free.[/green]",
        error_title="Remote Marathon could not start",
        ensure_local_tools=False,
    )


def run_dyno_dashboard() -> int:
    """Open Dyno directly without adding complexity to the normal launcher."""

    console = Console()
    models = discover_models()
    if not models:
        console.print("[bold red]No GGUF models found.[/bold red]")
        return 2
    selection = _initial_selection(models)
    try:
        tuned = _run_dyno_flow(console, selection)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dyno stopped. GPUs are free.[/yellow]")
        return 130
    except Exception as error:
        console.print(Panel(str(error), title="Dyno could not finish", border_style="red"))
        return 2
    if tuned:
        save_selection(tuned.model, tuned.profile, tuned.frontend)
    return 0
