// List of animal images
const animalImages = [
   
"leopard.jpg",
"elephant1.jpg",
"dog.jpg",
"mouse.jpg",
"Robot.png"
];

// Define feature data for each image
const animalFeatures = {
    "leopard.jpg": { tail: 1, whiskers: 0, trunk: 0, largeEars: 0 },
    "elephant1.jpg": { tail: 1, whiskers: 0, trunk: 1, largeEars: 1 },
    "dog.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 1 },
    "mouse.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 0 },
    "Robot.png": { tail: 0, whiskers: 0, trunk: 0, largeEars: 0 } // Robot as outlier
};

// Initialize the graph data
let graphData = [];

// Drag and Drop Functionality
const dropBox = document.getElementById("dropBox");
dropBox.addEventListener("dragover", (e) => e.preventDefault());
dropBox.addEventListener("drop", (e) => {
    e.preventDefault();
    const imageId = e.dataTransfer.getData("imageId");
    const imageElement = document.getElementById(imageId);

    if (imageElement) {
        dropBox.appendChild(imageElement.cloneNode(true));
        updateTable(imageId);
        updateGraph(animalFeatures[imageId]);
    }
});

// Update Table
function updateTable(imageId) {
    const table = document.getElementById("featureTable").querySelector("tbody");
    const { tail, whiskers, trunk, largeEars } = animalFeatures[imageId];

    const row = document.createElement("tr");
    row.innerHTML = `
        <td>${imageId}</td>
        <td>${tail}</td>
        <td>${whiskers}</td>
        <td>${trunk}</td>
        <td>${largeEars}</td>
    `;
    table.appendChild(row);
}

// Update Graph
function updateGraph(features) {
    const canvas = document.getElementById("featureGraph");
    const ctx = canvas.getContext("2d");

    // Add the new point
    const x = features.tail * 100 + features.whiskers * 50;
    const y = features.trunk * 100 + features.largeEars * 50;
    graphData.push({ x, y });

    // Clear and redraw the graph
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    graphData.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "blue";
        ctx.fill();
        ctx.closePath();
    });
}

// Make Animal Images Draggable
animalImages.forEach((imageName, index) => {
    const img = document.createElement("img");
    img.src = `images/${imageName}`; // Set the image source
    img.id = imageName; // Use filename as ID
    img.draggable = true;

    // Apply inline styles
    img.style.width = "100px"; // Adjust size
    img.style.height = "auto"; // Maintain aspect ratio
    img.style.margin = "10px"; // Add spacing between images
    img.style.cursor = "grab"; // Change cursor to indicate draggable

    img.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("imageId", imageName);
    });
    dropBox.parentNode.insertBefore(img, dropBox);
});
