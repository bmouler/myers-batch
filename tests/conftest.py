from hypothesis import settings

settings.register_profile(
    "det",
    settings(max_examples=200, derandomize=True, deadline=None),
)
settings.load_profile("det")
