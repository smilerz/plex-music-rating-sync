from typing import List, Optional, Union


class UserPrompt:
    def choice(
        self,
        message: str,
        options: List[str],
        *,
        default: Optional[Union[int, List[int]]] = None,
        allow_multiple: bool = False,
        help_text: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Prompt the user to select from a list of options."""
        while True:
            print("\n" + message)
            for idx, opt in enumerate(options, 1):
                print(f"  {idx}) {opt}")

            prompt = f"Select {'one or more' if allow_multiple else 'one'} [1-{len(options)}]"
            if help_text:
                prompt += " (type '?' for help)"
            prompt += ": "

            user_input = input(prompt).strip().lower()

            if user_input in {"?", "h", "help"}:
                if help_text:
                    print("\n" + help_text)
                else:
                    print("\n(No help available.)")
                continue

            try:
                if allow_multiple:
                    idxs = [int(i.strip()) for i in user_input.split(",")]
                    if all(1 <= i <= len(options) for i in idxs):
                        return [options[i - 1] for i in idxs]
                else:
                    idx = int(user_input)
                    if 1 <= idx <= len(options):
                        return options[idx - 1]
            except ValueError:
                pass

            print("Invalid input. Enter a number" + (" or comma-separated list" if allow_multiple else "") + ", or '?' for help.")

    def yes_no(self, message: str, *, default: bool = False) -> bool:
        """
        Prompt the user with a yes/no question.

        Args:
            message (str): The message to display.
            default (bool): What to return if the user just presses enter.

        Returns:
            bool: True for yes, False for no.
        """
        yes = {"y", "yes"}
        no = {"n", "no"}
        default_str = "[Y/n]" if default else "[y/N]"

        while True:
            choice = input(f"{message} {default_str} (type '?' for help): ").strip().lower()
            if choice in {"?", "h", "help"}:
                print("\nPlease enter 'y' for yes or 'n' for no.")
                continue
            if choice == "" and default is not None:
                return default
            elif choice in yes:
                return True
            elif choice in no:
                return False
            else:
                print("Please respond with 'y' or 'n'.")

    def text(self, message: str, *, validator=None, help_text=None) -> str:
        """
        Prompt the user for freeform input with optional validation.

        Args:
            message (str): The prompt to show.
            validator (Callable[[str], bool], optional): Input validation function.
            help_text (str, optional): Help message to show on '?' input.

        Returns:
            str: The validated user input.
        """
        full_prompt = f"{message}"
        if help_text:
            full_prompt += " (type '?' for help)"
        full_prompt += ": "

        while True:
            user_input = input(full_prompt).strip()

            if user_input in {"?", "h", "help"}:
                print("\n" + (help_text or "No help available."))
                continue

            if validator is None or validator(user_input):
                return user_input

            print("Invalid input. Try again or type '?' for help.")

    def confirm_continue(self, message: str = "Press Enter to continue or 'q' to quit: ") -> bool:
        response = input(message).strip().lower()
        return response != "q"
