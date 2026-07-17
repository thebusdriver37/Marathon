"""Arrow-key terminal workflow for selecting and running local models."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
    discover_models,
    find_model,
    find_profile,
    format_size,
    profiles_for_model,
    settings,
)
from .dyno import OBJECTIVES, candidate_profiles, run_tuning
from .frontends import direct_chat, run_codex
from .remote import (
    RemoteRuntime,
    fetch_remote_catalog,
    load_remote_selection,
    save_remote_selection,
)
from .runtime import Runtime, load_selection, save_selection


FRONTEND_NAMES = {"codex": "Codex", "direct": "Direct Chat"}


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
    if not warm and allow_tune:
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
        if action in {"codex", "direct"}:
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
        preferred = [item for item in models if item.family.id == "qwen3.6-27b"]
        model = preferred[0] if preferred else models[0]
    frontend = remembered.get("frontend", "codex")
    try:
        profile = find_profile(model, remembered.get("profile"), frontend)
    except ValueError:
        frontend = "codex"
        profile = find_profile(model, None, frontend)
    return Selection(model, profile, frontend)


def _launch_frontend(
    console: Console, runtime: Runtime | RemoteRuntime, frontend: str
) -> None:
    console.clear()
    if frontend == "direct":
        direct_chat(runtime, console)
        return
    code = run_codex(runtime)
    if code not in (0, 130):
        console.print(f"[yellow]Codex exited with status {code}.[/yellow]")


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
        console.print("[bold red]No GGUF models found.[/bold red]")
        console.print(f"Expected models under {settings().model_root}")
        return 2
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
    )


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
