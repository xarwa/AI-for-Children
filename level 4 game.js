// Feature data for each image (Updated)
const animalFeatures = {
    "bear1.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "bear2.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "Dog1.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 1, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "Dog2.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 1, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "Dog4.jpg": { tail: 1, whiskers: 1, trunk: 0, largeEars: 1, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "Robot.png": { tail: 0, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 0, feathers: 0, shell: 0 }, // Outlier
    "kangaroo.jpg": { tail: 1, whiskers: 0, trunk: 0, largeEars: 0, pouch: 1, talons: 0, wings: 0, beak: 0, fur: 1, feathers: 0, shell: 0 },
    "owl.jpg": { tail: 0, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 1, wings: 1, beak: 1, fur: 0, feathers: 1, shell: 0 },
    "Eagle.jpg": { tail: 0, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 1, wings: 1, beak: 1, fur: 0, feathers: 1, shell: 0 },
    "Parrot.jpg": { tail: 0, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 1, wings: 1, beak: 1, fur: 0, feathers: 1, shell: 0 },
    "turtle.jpg": { tail: 1, whiskers: 0, trunk: 0, largeEars: 0, pouch: 0, talons: 0, wings: 0, beak: 0, fur: 0, feathers: 0, shell: 1 }
};

// Cluster colors (Updated)
const clusterColors = {
    "bear1.jpg": "brown",
    "bear2.jpg": "brown",
    "Dog1.jpg": "green",
    "Dog2.jpg": "green",
    "Dog4.jpg": "green",
    "Robot.png": "gray",
    "kangaroo.jpg": "orange",
    "owl.jpg": "purple",
    "Eagle.jpg": "purple",
    "Parrot.jpg": "purple",
    "turtle.jpg": "teal"
};

// Graph data
let graphData = [];
let imageCounter = 1; // Image numbering
let droppedImages = new Set(); // To track already dropped images

// Normalizing features
function normalizeFeature(value, min, max) {
    return (value - min) / (max - min);
}

// Calculating dot position
function calculatePosition(features, clusterOffset = { x: 0, y: 0 }) {
    const featureValues = Object.values(features);
    const minFeature = Math.min(...featureValues);
    const maxFeature = Math.max(...featureValues);

    const normalized = featureValues.map(f => normalizeFeature(f, minFeature, maxFeature));
    const x = 50 + normalized[0] * 200 + clusterOffset.x; // Tail
    const y = 50 + normalized[1] * 200 + clusterOffset.y; // Whiskers

    return { x, y };
}

// Update graph
function updateGraph(imageId, features) {
    const canvas = document.getElementById("featureGraph");
    const ctx = canvas.getContext("2d");

    // Cluster-specific adjustments
    let clusterOffset = { x: 0, y: 0 };

    if (["Dog1.jpg", "Dog2.jpg", "Dog4.jpg"].includes(imageId)) {
        clusterOffset = { x: Math.random() * 20 - 10, y: Math.random() * 20 - 10 }; // Tight clustering for dogs
    } else if (["bear1.jpg", "bear2.jpg"].includes(imageId)) {
        clusterOffset = { x: Math.random() * 20 - 10, y: Math.random() * 20 - 10 }; // Tight clustering for bears
    } else if (["Eagle.jpg", "Parrot.jpg", "owl.jpg"].includes(imageId)) {
        clusterOffset = { x: Math.random() * 20 - 10, y: Math.random() * 20 - 10 }; // Tight clustering for birds
    } else if (imageId === "Robot.png") {
        clusterOffset = { x: 200, y: -200 }; // Outlier position
    } else if (imageId === "kangaroo.jpg" || imageId === "turtle.jpg") {
        clusterOffset = { x: Math.random() * 20 - 10, y: Math.random() * 20 - 10 }; // Separate kangaroo and turtle
    }

    const { x, y } = calculatePosition(features, clusterOffset);

    // Add point to graph data
    graphData.push({ x, y, color: clusterColors[imageId] });

    // Clear and redraw
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    graphData.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = point.color;
        ctx.fill();
        ctx.closePath();
    });
}

// Add image to table and graph
function updateTable(imageId) {
    const table = document.getElementById("featureTable").querySelector("tbody");
    const features = animalFeatures[imageId];

    const row = document.createElement("tr");
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

// Drag and drop logic
const dropBox = document.getElementById("dropBox");

dropBox.addEventListener("dragover", (e) => e.preventDefault());
dropBox.addEventListener("drop", (e) => {
    e.preventDefault();
    const imageId = e.dataTransfer.getData("imageId");

    // Check if the image has already been dropped
    if (!droppedImages.has(imageId)) {
        const imageElement = document.getElementById(imageId);

        if (imageElement) {
            dropBox.appendChild(imageElement.cloneNode(true)); // Clone the image
            droppedImages.add(imageId); // Mark as dropped
            updateTable(imageId); // Update table
            updateGraph(imageId, animalFeatures[imageId]); // Update graph
        }
    }
});

// Initialize draggable images
Object.keys(animalFeatures).forEach((imageName) => {
    const img = document.createElement("img");
    img.src = `images/${imageName}`;
    img.id = imageName;
    img.draggable = true;

    img.style.width = "200px";
    img.style.height = "auto";
    img.style.margin = "10px";
    img.style.cursor = "grab";

    img.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("imageId", imageName);
    });

    dropBox.parentNode.insertBefore(img, dropBox);
});

document.getElementById('nextPageButton').addEventListener('click', function() {
    window.location.href = 'Level 4 test 2 sound.html'; // Replace 'nextPage.html' with the URL of the next page
});
