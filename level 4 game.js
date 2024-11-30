// List of animal images (commented out in your original code)
/*
const animalImages = [
    "leopard.jpg",
    "elephant.jpg",
    "Dog1.jpg",
    "Dog2.jpg",
    "bear2.jpg",
    "kangaroo.jpg",
    "owl.jpg",
    "turtle.jpg",
    "Robot.png"
];
*/

// Define feature data for each image
const animalFeatures = {
    "leopard.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "elephant.jpg": { tail: 1, whiskers: 0, trunk: 1, largeEars: 1, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 0, feathers: 0, shell: 0 },
    "Dog1.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 1, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "Dog2.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 1, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "Robot.png": { tail: 0, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 0, feathers: 0, shell: 0 }, // Outlier
    "bear2.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "kangaroo.jpg": { tail: 1, whiskers: 0, trunk: 0, largeEars: 0, pouch: 1, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "owl.jpg": { tail: 0, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 1, wings: 1, beak: 1, fur: 0, feathers: 1, shell: 0 },
    "turtle.jpg": { tail: 1, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 0, feathers: 0, shell: 1 }
};

// Cluster colors for dots
const clusterColors = {
    "leopard.jpg": "red",
    "elephant.jpg": "blue",
    "Dog1.jpg": "green",
    "Dog2.jpg": "green",
    "Robot.png": "gray",
    "bear2.jpg": "brown",
    "kangaroo.jpg": "orange",
    "owl.jpg": "purple",
    "turtle.jpg": "teal"
};

// Initialize the graph data
let graphData = [];
let imageCounter = 1; // Start numbering images from 1

// Normalizing function
function normalizeFeature(featureValue, minValue, maxValue) {
    return (featureValue - minValue) / (maxValue - minValue);
}

// Mapping the features to a 2D position (canvas x, y)
function calculatePosition(features) {
    const featureValues = Object.values(features);
    const minFeature = Math.min(...featureValues);
    const maxFeature = Math.max(...featureValues);

    const normalizedFeatures = featureValues.map(f => normalizeFeature(f, minFeature, maxFeature));

    const x = 50 + normalizedFeatures[0] * 200;  // Tail
    const y = 50 + normalizedFeatures[1] * 200;  // Whiskers

    return { x, y };
}

// Update Graph
function updateGraph(imageId, features) {
    const canvas = document.getElementById("featureGraph");
    const ctx = canvas.getContext("2d");

    // Calculate dot position based on normalized features
    const { x, y } = calculatePosition(features);

    // Add the new point to graphData
    graphData.push({ x, y, color: clusterColors[imageId] });

    // Clear and redraw the graph
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    graphData.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 10, 0, Math.PI * 2); // Draw a circle with radius 10
        ctx.fillStyle = point.color; // Assign color based on cluster
        ctx.fill();
        ctx.closePath();
    });
}

// Make Animal Images Draggable
const dropBox = document.getElementById("dropBox");

dropBox.addEventListener("dragover", (e) => e.preventDefault());
dropBox.addEventListener("drop", (e) => {
    e.preventDefault();
    const imageId = e.dataTransfer.getData("imageId");
    const imageElement = document.getElementById(imageId);

    if (imageElement) {
        dropBox.appendChild(imageElement.cloneNode(true));
        updateTable(imageId);
        updateGraph(imageId, animalFeatures[imageId]);
    }
});

// Update Table with Numbered Image
function updateTable(imageId) {
    const table = document.getElementById("featureTable").querySelector("tbody");
    const features = animalFeatures[imageId];

    const row = document.createElement("tr");

    // Assign image number in the first column instead of image name
    row.innerHTML = `
        <td>${imageCounter++}</td>
        <td>${features.tail}</td>
        <td>${features.whiskers}</td>
        <td>${features.trunk}</td>
        <td>${features.largeEars}</td>
        <td>${features.pouch}</td>
        <td>${features.talons}</td>
        <td>${features.wings}</td>
        <td>${features.beak}</td>
        <td>${features.fur}</td>
        <td>${features.feathers}</td>
        <td>${features.shell}</td>
    `;

    table.appendChild(row);
}

// Make animal images draggable
const animalImages = Object.keys(animalFeatures);
animalImages.forEach((imageName) => {
    const img = document.createElement("img");
    img.src = `images/${imageName}`; // Set the image source
    img.id = imageName; // Use filename as ID
    img.draggable = true;

    img.style.width = "200px"; // Adjust size
    img.style.height = "auto"; // Maintain aspect ratio
    img.style.margin = "10px"; // Add spacing between images
    img.style.cursor = "grab"; // Change cursor to indicate draggable

    img.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("imageId", imageName);
    });
    dropBox.parentNode.insertBefore(img, dropBox);
});
