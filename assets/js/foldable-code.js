document.addEventListener("DOMContentLoaded", function () {
    const codeBlocks = document.querySelectorAll("pre > code");
    codeBlocks.forEach((block) => {
        const pre = block.parentNode;

        // Set a max-height for foldable blocks
        const maxHeight = 600; // Adjust as needed
        if (block.offsetHeight > maxHeight) {
            pre.style.maxHeight = `${maxHeight}px`;
            pre.style.overflow = "hidden";
            pre.style.position = "relative";

            const toggleButton = document.createElement("button");
            toggleButton.innerText = "More";
            toggleButton.className = "fold-button";

            toggleButton.addEventListener("click", () => {
                if (pre.style.maxHeight) {
                    pre.style.maxHeight = "";
                    pre.style.overflow = "visible";
                    toggleButton.innerText = "Less";
                } else {
                    pre.style.maxHeight = `${maxHeight}px`;
                    pre.style.overflow = "hidden";
                    toggleButton.innerText = "More";
                }
            });

            pre.appendChild(toggleButton);
        }
    });
});
