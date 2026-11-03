"""Explicit continuation of an open assistant message in an Instruct model."""
from stegtext.v2.model.prompt import ChatPromptPolicy


class AssistantPrefillPromptPolicy(ChatPromptPolicy):
    def __init__(self, assistant_prefix, *, template_options=None):
        super().__init__(template_options=template_options)
        if not isinstance(assistant_prefix, str) or not assistant_prefix:
            raise ValueError('Assistant prefill must be a nonempty string')
        self.assistant_prefix = assistant_prefix

    def render(self, tokenizer, prompt):
        return self._render(tokenizer, prompt, tokenize=True, return_tensors='pt')

    def render_text(self, tokenizer, prompt):
        return self._render(tokenizer, prompt, tokenize=False)

    def _render(self, tokenizer, prompt, **options):
        return tokenizer.apply_chat_template(
            [dict(role='user', content=str(prompt)),
             dict(role='assistant', content=self.assistant_prefix)],
            add_generation_prompt=False, continue_final_message=True,
            **options, **self.template_options)


def configure_prompt(source, assistant_prefix=None):
    """Set the explicit public prompt context before hashing, sampling or replay."""
    configure = getattr(source, 'configure_assistant_prefix', None)
    if configure is not None:
        configure(assistant_prefix)
    elif assistant_prefix is not None:
        raise ValueError('Source does not support assistant prefill')
