from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_assistant_markdown_renderer_converts_bold_and_escapes_html() -> None:
    app_js = Path(__file__).parents[1] / "app" / "static" / "app.js"
    source = app_js.read_text(encoding="utf-8")
    functions = source[source.index("function escapeHtml") : source.index("function formatTime")]
    sample = (
        "**优点: **\n"
        "- **频谱效率提升**: <script>alert(1)</script>\n"
        "\n"
        "普通段落 `code`"
    )
    script = (
        functions
        + "\n"
        + f"const output = renderMarkdown({json.dumps(sample, ensure_ascii=False)});\n"
        + "if (output.includes('**')) throw new Error(output);\n"
        + "if (!output.includes('<strong>优点:</strong>')) throw new Error(output);\n"
        + "if (!output.includes('<li><strong>频谱效率提升</strong>: &lt;script&gt;alert(1)&lt;/script&gt;</li>')) throw new Error(output);\n"
        + "if (!output.includes('<code>code</code>')) throw new Error(output);\n"
    )

    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr or result.stdout
