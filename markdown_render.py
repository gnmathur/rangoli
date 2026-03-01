"""Markdown-to-tkinter text widget rendering."""

import re

MD_HEADER_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
MD_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
MD_ITALIC_RE = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)')
MD_BULLET_RE = re.compile(r'^(\s*)[-*+]\s+', re.MULTILINE)
MD_NUMBERED_RE = re.compile(r'^(\s*)\d+\.\s+', re.MULTILINE)


def insert_markdown(textbox, md_text):
    """Insert markdown text into a CTkTextbox with basic formatting tags.

    Supports: headers (bold, larger), **bold**, *italic*, bullet/numbered lists.
    """
    tb = textbox._textbox  # underlying tk.Text widget

    # Configure tags
    tb.tag_configure("h1", font=("Menlo", 18, "bold"), spacing3=6)
    tb.tag_configure("h2", font=("Menlo", 16, "bold"), spacing3=5)
    tb.tag_configure("h3", font=("Menlo", 14, "bold"), spacing3=4)
    tb.tag_configure("bold", font=("Menlo", 13, "bold"))
    tb.tag_configure("italic", font=("Menlo", 13, "italic"))

    lines = md_text.split("\n")
    for i, line in enumerate(lines):
        if i > 0:
            tb.insert("end", "\n")

        # Check for header
        hm = MD_HEADER_RE.match(line)
        if hm:
            level = min(len(hm.group(1)), 3)
            tb.insert("end", hm.group(2), f"h{level}")
            continue

        # Convert bullet markers to a clean bullet character
        line = MD_BULLET_RE.sub(lambda m: m.group(1) + "\u2022 ", line)
        # Keep numbered list formatting as-is (already readable)

        # Parse inline bold/italic spans
        # Process **bold** first
        result_parts = []
        last_end = 0
        for bm in MD_BOLD_RE.finditer(line):
            if bm.start() > last_end:
                result_parts.append((line[last_end:bm.start()], None))
            result_parts.append((bm.group(1), "bold"))
            last_end = bm.end()
        if last_end < len(line):
            result_parts.append((line[last_end:], None))

        if not result_parts:
            result_parts = [(line, None)]

        # Process *italic* within non-bold segments
        final_parts = []
        for text, tag in result_parts:
            if tag is not None:
                final_parts.append((text, tag))
                continue
            last_end = 0
            for im in MD_ITALIC_RE.finditer(text):
                if im.start() > last_end:
                    final_parts.append((text[last_end:im.start()], None))
                final_parts.append((im.group(1), "italic"))
                last_end = im.end()
            if last_end < len(text):
                final_parts.append((text[last_end:], None))

        for text, tag in final_parts:
            if tag:
                tb.insert("end", text, tag)
            else:
                tb.insert("end", text)
