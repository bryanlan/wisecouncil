# utils.py

import re
import json
from langgraph.graph import Graph, END
 

def clean_response(content: str) -> str:
    """
    Cleans the response content by:
    - Removing code fences and JSON labels
    - Removing outer quotes if present
    - Fixing invalid escape sequences
    - Ensuring the string can be parsed as valid JSON
    """
    if content is None or content == '':
        return ''
    # Remove code fences and JSON labels
    content = content.strip()
    if content.startswith('```'):
        content = content.strip('`')
        content = re.sub(r'^json\n', '', content, flags=re.IGNORECASE).strip()

    # Remove outer quotes if present
    if (content.startswith("'") and content.endswith("'")) or \
       (content.startswith('"') and content.endswith('"')):
        content = content[1:-1]

    # Replace invalid escape sequences
    # Replace \' with '
    content = content.replace("\\'", "'")

    # Remove any remaining backslashes not part of valid escape sequences
    # Valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # We'll use a regex to find invalid ones and remove the backslash
    def fix_invalid_escapes(match):
        escape = match.group(0)
        if re.match(r'\\u[0-9a-fA-F]{4}', escape):
            return escape  # Valid Unicode escape sequence
        elif escape in ('\\"', '\\\\', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t'):
            return escape  # Valid escape sequence
        else:
            return escape[1:]  # Remove the backslash

    content = re.sub(r'\\.', fix_invalid_escapes, content)

    # Remove trailing commas before closing braces or brackets
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Now, try to load the JSON to check if it's valid
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding failed: {e.msg} at line {e.lineno} column {e.colno} (char {e.pos})")

    return content

def format_messages(messages, suppressToolMsg=True) -> str:
    TOOLPREFIX = "Tool Provided Data:"
    if suppressToolMsg:
        return "\n\n".join([f"{m.content}" for m in messages if not m.content.startswith(TOOLPREFIX)])
    else:
        return "\n\n".join([f"{m.content}" for m in messages])

def should_continue(state):
    nextState = state.get('nextAgent', 'END')
    if nextState.upper() == 'END':
        nextState = END
    if nextState is None or nextState =='none':
        ProblemState = True
        nextState = END

    return nextState
