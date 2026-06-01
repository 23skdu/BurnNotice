import re

with open("arxiv/paper.tex", "r") as f:
    content = f.read()

# Replace email
content = re.sub(
    r"\\author\{\s*Ross Dmochowski \\\\\s*\\texttt\{rsd@rsd\.dev\} \\\\\s*\}",
    r"\\author{\n  Ross Dmochowski\n}",
    content
)

# Replace code blocks with links
blocks = [
    (r"\\begin\{lstlisting\}\[language=Go, caption=ONNX model loader with forward hook registration\].*?\\end\{lstlisting\}",
     r"The full source code for the ONNX model loader with forward hook registration can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/engine}."),
    
    (r"\\begin\{lstlisting\}\[language=Go, caption=WASM inference backend\].*?\\end\{lstlisting\}",
     r"The full source code for the WASM inference backend can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/wasm}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=Activation scoring with gonum linear algebra\].*?\\end\{lstlisting\}",
     r"The full source code for activation scoring with \\texttt{gonum} linear algebra can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/scoring}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=Steering intervention implementations\].*?\\end\{lstlisting\}",
     r"The full source code for steering intervention implementations can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/steer}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=Profile serialization and versioning\].*?\\end\{lstlisting\}",
     r"The full source code for profile serialization and versioning can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/profile}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=CLI tool for HuggingFace model download and ONNX conversion\].*?\\end\{lstlisting\}",
     r"The full source code for the CLI tool handling HuggingFace model downloads and ONNX conversions can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/cmd/burnnotice}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=HuggingFace model downloader\].*?\\end\{lstlisting\}",
     r"The full source code for the HuggingFace model downloader can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/downloader}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=Activation buffer pooling\].*?\\end\{lstlisting\}",
     r"The full source code for activation buffer pooling can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/memory}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=Checkpoint manager\].*?\\end\{lstlisting\}",
     r"The full source code for the checkpoint manager can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/checkpoint}."),
     
    (r"\\begin\{lstlisting\}\[language=Go, caption=SSE event hub for real-time dashboard updates\].*?\\end\{lstlisting\}",
     r"The full source code for the SSE event hub driving real-time dashboard updates can be found in the GitHub repository at \\url{https://github.com/rsd/BurnNotice/tree/main/internal/web}.")
]

for pattern, repl in blocks:
    content = re.sub(pattern, repl, content, flags=re.DOTALL)

with open("arxiv/paper.tex", "w") as f:
    f.write(content)

print("Updated paper.tex")
