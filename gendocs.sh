#!/bin/bash

pdoc overreact \
  --docformat "numpy" \
  --edit-url "overreact=https://github.com/geem-lab/overreact/blob/main/overreact/" \
  --footer-text "overreact" \
  --logo "https://raw.githubusercontent.com/geem-lab/overreact-guide/master/logo.png" \
  --logo-link "/overreact" \
  --math \
  --search \
  --show-source
# --output-directory docs/
