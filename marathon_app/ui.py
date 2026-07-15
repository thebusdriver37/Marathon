"""Arrow-key terminal workflow for selecting and running local models."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .catalog import (
    Model,
    Profile,
    discover_models,
    find_model,
    find_profile,
    format_size,
    settings,
)
from .frontends import direct_chat, run_codex
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
            f"Central GGUF · {len(model.family.profiles)} runtime profiles",
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
        for profile in model.family.profiles
    ]


def _choose_model_profile(
    console: Console, models: list[Model], selection: Selection
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
            context=("Central model library", str(settings().model_root)),
        )
        if chosen_model is None:
            return None
        model = models[chosen_model]
        model_index = chosen_model
        profiles = list(model.family.profiles)
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


def _home_items(selection: Selection, *, warm: bool) -> list[MenuItem]:
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
) -> tuple[str, Selection]:
    while True:
        items = _home_items(selection, warm=warm)
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
            changed = _choose_model_profile(console, models, selection)
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


def _initial_selection(models: list[Model]) -> Selection:
    remembered = load_selection()
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


def _launch_frontend(console: Console, runtime: Runtime, frontend: str) -> None:
    console.clear()
    if frontend == "direct":
        direct_chat(runtime, console)
        return
    code = run_codex(runtime)
    if code not in (0, 130):
        console.print(f"[yellow]Codex exited with status {code}.[/yellow]")


def run_dashboard(initial_frontend: str | None = None) -> int:
    console = Console()
    models = discover_models()
    if not models:
        console.print("[bold red]No GGUF models found.[/bold red]")
        console.print(f"Expected models under {settings().model_root}")
        return 2
    selection = _initial_selection(models)
    if initial_frontend:
        try:
            selection.profile = find_profile(selection.model, selection.profile.id, initial_frontend)
        except ValueError:
            selection.profile = find_profile(selection.model, None, initial_frontend)
        selection.frontend = initial_frontend

    preferred = initial_frontend
    while True:
        try:
            action, selection = _home(
                console,
                models,
                selection,
                warm=False,
                preferred_frontend=preferred,
            )
        except KeyboardInterrupt:
            console.print()
            return 130
        preferred = None
        if action == "quit":
            return 0
        save_selection(selection.model, selection.profile, action)
        runtime = Runtime(selection.model, selection.profile)
        try:
            with console.status("[bold magenta]Preparing GPUs…[/bold magenta]", spinner="dots") as status:
                runtime.start(lambda message: status.update(f"[magenta]{message}[/magenta]"))
            _launch_frontend(console, runtime, action)
            while True:
                warm_action, selection = _home(console, models, selection, warm=True)
                if warm_action == "quit":
                    action = "quit"
                    break
                if warm_action == "change":
                    action = "switch"
                    break
                save_selection(selection.model, selection.profile, warm_action)
                _launch_frontend(console, runtime, warm_action)
        except KeyboardInterrupt:
            action = "quit"
        except Exception as error:
            console.print(Panel(str(error), title="Marathon could not start", border_style="red"))
            Prompt.ask("Press Enter to return", default="")
            action = "switch"
        finally:
            with console.status("[yellow]Stopping backend and freeing GPUs…[/yellow]", spinner="dots"):
                runtime.cleanup()
        if action == "quit":
            console.print("[green]Backend stopped. GPUs are free.[/green]")
            return 0
