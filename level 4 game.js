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
const clusters = document.querySelectorAll('.cluster');
clusters.forEach(cluster => {
    cluster.addEventListener('dragover', allowDrop);
    cluster.addEventListener('drop', drop);
});

function allowDrop(event) {
    event.preventDefault();
}

// Drop event
function drop(event) {
    event.preventDefault();

    const animalType = event.dataTransfer.getData('animalType');
    const clusterType = event.currentTarget.dataset.cluster;
    const imageSrc = event.dataTransfer.getData('imageSrc');

    // Debugging logs
    console.log("Dropped:", animalType);
    console.log("Cluster Type:", clusterType);

    // Check if the dropped animal matches the cluster type
    if (animalType === clusterType) {
        const draggedElement = document.querySelector(`.draggable.dragging[src="${imageSrc}"]`);
        if (draggedElement) {
            // Remove the 'dragging' class and append to the correct cluster
            draggedElement.classList.remove('dragging');
            event.currentTarget.appendChild(draggedElement);
            draggedElement.style.position = 'static'; // Reset positioning within the container
            draggedElement.style.top = ''; // Remove inline positioning
            draggedElement.style.left = ''; // Remove inline positioning

            // Provide feedback
            setTimeout(() => {
                alert(`Good job! You helped the ${animalType} family.`);
            }, 300);
        }
    } else {
        alert("Oops! This animal doesn't belong here. Try again.");
    }
}

// Initialize the game by scattering animals
scatterAnimals();
