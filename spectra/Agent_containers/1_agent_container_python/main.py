import code
from fastapi import FastAPI, Request, HTTPException
from typing import Dict, Optional
import io
import contextlib
import json
import warnings
import asyncio
import signal
import sys
import resource
import os
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

app = FastAPI()

MAX_EXECUTION_TIME = 60  # seconds
MAX_MEMORY_MB = 1024  # MB
SAFE_MODULES = {"numpy", "scipy", "sympy", "math", "random", "datetime", "time"}

def setup_security_limits():
    """Set resource limits for the process"""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_MB * 1024 * 1024, MAX_MEMORY_MB * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_EXECUTION_TIME, MAX_EXECUTION_TIME))
    except (ValueError, OSError):
        pass

def validate_code(code: str) -> bool:
    """Basic validation to block obviously dangerous patterns"""
    dangerous_keywords = [
        'import os', 'import sys', 'import subprocess', 'import importlib',
        'from os', 'from sys', 'from subprocess', 'from importlib',
        '__import__', 'exec(', 'eval(', 'open(', 'file(',
        'globals(', 'locals(', 'vars(', 'dir(',
        'getattr', 'setattr', 'delattr', 'hasattr'
    ]
    code_lower = code.lower()
    for keyword in dangerous_keywords:
        if keyword in code_lower:
            return False
    return True

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Allow only whitelisted modules"""
    if name in SAFE_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is not allowed")

async def execute_with_timeout(code: str, timeout: int = MAX_EXECUTION_TIME) -> tuple[bool, str]:
    """Execute code with timeout and security restrictions"""
    try:
        # Validate code first
        if not validate_code(code):
            return False, "Code contains restricted operations"
        
        # Setup security limits
        setup_security_limits()
        
        # Restricted globals
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'list': list,
                'dict': dict,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'tuple': tuple,
                'set': set,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                '__import__': safe_import,
            }
        }
        
        f = io.StringIO()
        
        async def run_code():
            with contextlib.redirect_stdout(f):
                exec(code, safe_globals, {})
            return f.getvalue().strip()
        
        try:
            output = await asyncio.wait_for(run_code(), timeout=timeout)
            return True, output
        except asyncio.TimeoutError:
            return False, f"Code execution timed out after {timeout} seconds"
            
    except Exception as e:
        return False, f"Execution error: {str(e)}"

@app.post("/run_python")
async def run_python_code(request: Request):
    """
    Executes Python code with security restrictions.
    Only returns 'success' and 'message'.
    """
    try:
        body_bytes = await request.body()
        
        if not body_bytes:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        try:
            request_data = json.loads(body_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        code = None
        
        if "arguments" in request_data:
            arguments = request_data["arguments"]
            if isinstance(arguments, str):
                try:
                    unescaped_string = json.loads(arguments)
                    if isinstance(unescaped_string, str) and unescaped_string.startswith("code:"):
                        code = unescaped_string[len("code:"):]
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(arguments, dict):
                code = arguments.get("code")
        elif "code" in request_data:
            code = request_data["code"]
        
        if not code:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract code from request. Expected 'code' field or 'arguments.code'"
            )
        
        if len(code) > 10000:  # Limit of code size
            raise HTTPException(status_code=400, detail="Code too long (max 10,000 characters)")
        
        success, output = await execute_with_timeout(code)

        return {
            "success": str(success),
            "message": output
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Azure"""
    return {"status": "healthy", "service": "python-executor"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
