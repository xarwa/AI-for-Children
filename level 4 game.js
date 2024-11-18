// Select the container to hold scattered animals
const animalsContainer = document.getElementById('animals');

// Array of animal images for each type
const animalImages = {
    lion: ['cub1.png', 'cub2.png', 'cub3.png', 'cub4.png', 'cub5.png', 'cub6.png'],
    leopard: ['leopard-baby1.jpg', 'leopard-baby2.jpg', 'leopard-baby3.jpg', 'leopard-baby4.jpg', 'leopard-baby5.jpg', 'leopard-baby6.jpg', 'leopard-baby7.jpg', 'leopard-baby8.jpg'],
    blackLeopard: ['black-leopard-baby1.jpg', 'black-leopard-baby2.jpg'],
    tiger: ['tiger-baby1.jpg', 'tiger-baby2.jpg']
};

// Generate unique random positions for initial layout
function getRandomPosition(existingPositions, maxWidth, maxHeight) {
    let position;
    do {
        position = {
            top: Math.floor(Math.random() * (maxHeight - 80)),
            left: Math.floor(Math.random() * (maxWidth - 80))
        };
    } while (existingPositions.some(pos => Math.abs(pos.top - position.top) < 60 && Math.abs(pos.left - position.left) < 60));
    existingPositions.push(position);
    return position;
}

// Function to initialize and scatter animals without overlap
function scatterAnimals() {
    const existingPositions = [];
    const maxWidth = window.innerWidth;
    const maxHeight = window.innerHeight;

    Object.keys(animalImages).forEach(animalType => {
        animalImages[animalType].forEach(image => {
            const imgElement = document.createElement('img');
            imgElement.src = image;
            imgElement.alt = `${animalType} baby`;
            imgElement.classList.add('draggable');
            imgElement.draggable = true;
            imgElement.dataset.animal = animalType;

            // Generate non-overlapping random position
            const position = getRandomPosition(existingPositions, maxWidth, maxHeight);
            imgElement.style.top = `${position.top}px`;
            imgElement.style.left = `${position.left}px`;
            imgElement.style.position = 'absolute';

            // Add event listeners for dragging
            imgElement.addEventListener('dragstart', dragStart);

            // Append to the main document body (not in a container)
            document.body.appendChild(imgElement);
        });
    });
}

// Drag start event
function dragStart(event) {
    event.dataTransfer.setData('animalType', event.target.dataset.animal);
    event.dataTransfer.setData('imageSrc', event.target.src); // Store image source for identification
    event.target.classList.add('dragging'); // Add a class for identifying dragged element
    console.log("Dragging:", event.target.alt); // Debugging log
}

// Allow drop functionality only on clusters
// Game Data
const animalTypes = ["lion", "elephant", "giraffe", "zebra"];
const hints = [
    "Hint: Look for animals with manes!",
    "Hint: Look for animals with trunks!",
    "Hint: Look for animals with long necks!",
    "Hint: Look for animals with stripes!"
];
const correctClusters = {
    lion: "cluster1",
    elephant: "cluster2",
    giraffe: "cluster3",
    zebra: "cluster1",
};

// DOM Elements
const forest = document.getElementById("forest");
const hintElement = document.getElementById("hint");
const feedbackElement = document.getElementById("feedback");
const checkBtn = document.getElementById("check-btn");

// Generate Random Animals
function createAnimals() {
    forest.innerHTML = ""; // Clear previous round

    for (let i = 0; i < 10; i++) {
        const type = animalTypes[Math.floor(Math.random() * animalTypes.length)];
        const animal = document.createElement("img");
        animal.src = `images/${type}.png`; // Use your image paths
        animal.classList.add("animal");
        animal.setAttribute("data-family", type);

        // Random Positioning
        animal.style.top = `${Math.random() * 80}%`;
        animal.style.left = `${Math.random() * 80}%`;

        // Make Draggable
        animal.draggable = true;
        animal.addEventListener("dragstart", dragStart);

        forest.appendChild(animal);
    }
}

// Drag-and-Drop Handlers
function dragStart(event) {
    event.dataTransfer.setData("animalType", event.target.getAttribute("data-family"));
    event.dataTransfer.setData("animalID", event.target.id);
}

const clusters = document.querySelectorAll(".cluster");
clusters.forEach(cluster => {
    cluster.addEventListener("dragover", event => event.preventDefault());
    cluster.addEventListener("drop", dropAnimal);
});

function dropAnimal(event) {
    const animalType = event.dataTransfer.getData("animalType");
    const animal = document.querySelector(`[data-family='${animalType}']`);
    event.target.appendChild(animal);
}

// Show Hint
function showHint() {
    const randomHint = hints[Math.floor(Math.random() * hints.length)];
    hintElement.textContent = randomHint;
}

// Check Clusters
function checkClusters() {
    const cluster1 = document.querySelectorAll("#cluster1 img");
    const cluster2 = document.querySelectorAll("#cluster2 img");
    const cluster3 = document.querySelectorAll("#cluster3 img");

    const clusters = { cluster1, cluster2, cluster3 };

    let correct = true;

    for (const [type, clusterID] of Object.entries(correctClusters)) {
        const cluster = clusters[clusterID];
        const inCluster = Array.from(cluster).some(animal => animal.getAttribute("data-family") === type);
        if (!inCluster) correct = false;
    }

    feedbackElement.textContent = correct
        ? "Great job! All families are correctly grouped!"
        : "Oops! Some animals are in the wrong groups.";
}

// Event Listeners
checkBtn.addEventListener("click", checkClusters);

// Initialize Game
createAnimals();
showHint();
