document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("div[class^='language-']").forEach(block => {
        const languageClass = block.className.match(/language-(\w+)/);
        const language = languageClass ? languageClass[1] : "code";

        const wrapper = document.createElement("div");
        wrapper.classList.add("code-block-wrapper");

        const header = document.createElement("div");
        header.classList.add("code-block-header");
        header.innerHTML = `<span class="language-label">${language}</span>
                            <button class="copy-button" onclick="copyToClipboard(this)">Copy</button>`;

        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(header);
        wrapper.appendChild(block);
    });
});

function copyToClipboard(button) {
    const codeBlock = button.closest(".code-block-wrapper").querySelector("code");
    navigator.clipboard.writeText(codeBlock.innerText).then(() => {
        button.innerText = "Copied!";
        setTimeout(() => (button.innerText = "Copy"), 2000);
    }).catch(() => {
        button.innerText = "Error";
        setTimeout(() => (button.innerText = "Copy"), 2000);
    });
}
