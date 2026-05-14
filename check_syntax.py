import ast

try:
    with open('src/agents/agent.py', 'r') as f:
        code = f.read()
    ast.parse(code)
    print('File parses successfully')
except SyntaxError as e:
    print(f'Syntax error at line {e.lineno}: {e.msg}')
    lines = code.split('\n')
    
    # Find unclosed docstrings
    quote_count = 0
    for i, line in enumerate(lines[:e.lineno], start=1):
        count = line.count('"""')
        if count > 0:
            quote_count += count
            print(f'Line {i} ({quote_count} total): {line.strip()[:100]}')
