/**
 * Political Latent Space - Entity Focus View
 * 
 * This file contains the code for the 2D entity focus visualization
 * that shows relationships between political entities.
 */

// Main function to create the 2D focus visualization
function createFocusVisualization(data, containerId) {
    // Container for the visualization
    const container = document.getElementById(containerId);
    
    // Extract data
    const focusEntity = data.entity;
    const relatedEntities = data.related_entities;
    const wordCloud = data.word_cloud;
    
    // Set up the visualization dimensions
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Create a force-directed graph layout
    const nodes = [];
    const links = [];
    
    // Add the focus entity as the central node
    nodes.push({
        id: `${focusEntity.type}-${focusEntity.name}`,
        name: focusEntity.name,
        type: focusEntity.type,
        size: 30, // Larger size for the focus entity
        color: focusEntity.type === 'movement' ? '#e63946' : '#457b9d', // Red for movements, blue for politicians
        x: width / 2,
        y: height / 2,
        fixed: true // Keep the focus entity fixed at the center
    });
    
    // Add related entities as nodes
    relatedEntities.forEach((entity, i) => {
        // Calculate position in a circle around the center
        const angle = (i / relatedEntities.length) * 2 * Math.PI;
        const radius = 200; // Distance from center
        const x = width / 2 + radius * Math.cos(angle);
        const y = height / 2 + radius * Math.sin(angle);
        
        // Determine node size based on similarity
        const size = 10 + entity.similarity * 15;
        
        // Determine color based on type
        let color;
        if (entity.type === 'movement') {
            // Gradient from yellow to red based on similarity
            const hue = 60 - entity.similarity * 60; // 60 is yellow, 0 is red
            color = `hsl(${hue}, 90%, 50%)`;
        } else {
            // Gradient from light blue to dark blue based on similarity
            const lightness = 70 - entity.similarity * 30; // Lighter blue for less similar
            color = `hsl(210, 90%, ${lightness}%)`;
        }
        
        // Add the node
        nodes.push({
            id: `${entity.type}-${entity.name}`,
            name: entity.name,
            type: entity.type,
            movement: entity.movement || null,
            size: size,
            color: color,
            x: x,
            y: y,
            similarity: entity.similarity
        });
        
        // Add a link from the focus entity to this entity
        links.push({
            source: `${focusEntity.type}-${focusEntity.name}`,
            target: `${entity.type}-${entity.name}`,
            value: entity.similarity * 5, // Link thickness based on similarity
            color: entity.type === 'movement' ? 'rgba(255, 100, 100, 0.3)' : 'rgba(100, 100, 255, 0.3)'
        });
    });
    
    // Add word cloud terms as smaller nodes
    wordCloud.slice(0, 20).forEach((term, i) => {
        // Calculate position in an outer circle
        const angle = (i / 20) * 2 * Math.PI;
        const radius = 350; // Further from center than entities
        const x = width / 2 + radius * Math.cos(angle);
        const y = height / 2 + radius * Math.sin(angle);
        
        // Determine node size based on term weight
        const size = 5 + term.value * 10;
        
        // Determine color based on position values
        const r = Math.abs(term.position[0]) * 255;
        const g = Math.abs(term.position[2]) * 255;
        const b = Math.abs(term.position[1]) * 255;
        const color = `rgba(${r}, ${g}, ${b}, 0.8)`;
        
        // Add the node
        nodes.push({
            id: `term-${term.text}`,
            name: term.text,
            type: 'term',
            size: size,
            color: color,
            x: x,
            y: y,
            value: term.value
        });
        
        // Add a thin link from the focus entity to this term
        links.push({
            source: `${focusEntity.type}-${focusEntity.name}`,
            target: `term-${term.text}`,
            value: 1, // Thin links for terms
            color: 'rgba(200, 200, 200, 0.1)'
        });
    });
    
    // Create the network visualization using Plotly
    const trace = {
        x: nodes.map(node => node.x),
        y: nodes.map(node => node.y),
        mode: 'markers+text',
        marker: {
            size: nodes.map(node => node.size),
            color: nodes.map(node => node.color),
            line: { width: 1, color: '#333' }
        },
        text: nodes.map(node => node.name),
        textposition: 'bottom center',
        textfont: {
            family: 'Arial',
            size: nodes.map(node => node.type === 'term' ? 10 : 14),
            color: nodes.map(node => node.type === 'term' ? '#ddd' : '#fff')
        },
        hoverinfo: 'text',
        hovertext: nodes.map(node => {
            if (node.type === 'movement') {
                return `Movement: ${node.name}`;
            } else if (node.type === 'politician') {
                return `Politician: ${node.name}<br>Movement: ${node.movement || 'Unknown'}`;
            } else {
                return `Term: ${node.name}<br>Weight: ${node.value.toFixed(2)}`;
            }
        }),
        ids: nodes.map(node => node.id)
    };
    
    // Create edges as shapes
    const shapes = links.map(link => {
        const sourceNode = nodes.find(node => node.id === link.source);
        const targetNode = nodes.find(node => node.id === link.target);
        
        return {
            type: 'line',
            x0: sourceNode.x,
            y0: sourceNode.y,
            x1: targetNode.x,
            y1: targetNode.y,
            line: {
                color: link.color,
                width: link.value
            },
            layer: 'below'
        };
    });
    
    // Layout configuration
    const layout = {
        showlegend: false,
        hovermode: 'closest',
        margin: { l: 20, r: 20, b: 20, t: 50 },
        xaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            range: [0, width]
        },
        yaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            range: [0, height]
        },
        shapes: shapes,
        title: {
            text: `${focusEntity.name} - Relationship Network`,
            font: {
                family: 'Arial',
                size: 24,
                color: '#fff'
            }
        },
        paper_bgcolor: '#1a1a1a',
        plot_bgcolor: '#1a1a1a',
        annotations: [{
            text: 'Click and drag to move, scroll to zoom',
            showarrow: false,
            x: 0.5,
            y: 1.05,
            xref: 'paper',
            yref: 'paper',
            font: {
                family: 'Arial',
                size: 12,
                color: '#aaa'
            }
        }]
    };
    
    // Create the plot
    Plotly.newPlot(containerId, [trace], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toggleSpikelines'],
        displaylogo: false
    });
    
    // Add click event to nodes
    document.getElementById(containerId).on('plotly_click', function(data) {
        const pointIndex = data.points[0].pointIndex;
        const clickedNode = nodes[pointIndex];
        
        // Skip if clicked on the focus entity or a term
        if (clickedNode.id === `${focusEntity.type}-${focusEntity.name}` || clickedNode.type === 'term') {
            return;
        }
        
        // Navigate to a new focus view for the clicked entity
        loadEntityFocusView(clickedNode.type, clickedNode.name);
    });
}

// Function to load entity focus view
function loadEntityFocusView(entityType, entityName) {
    // Show loading indicator
    document.getElementById('focus-visualization').innerHTML = '<div class="loading">Loading...</div>';
    
    // Update the title
    document.getElementById('focus-entity-title').textContent = entityName;
    
    // Hide galaxy view, show focus view
    document.getElementById('galaxy-container').style.display = 'none';
    document.getElementById('focus-container').style.display = 'block';
    
    // Fetch the entity focus data
    fetch(`/api/entity-focus?entity_type=${entityType}&entity_name=${encodeURIComponent(entityName)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('focus-visualization').innerHTML = 
                    `<div class="error">Error: ${data.message}</div>`;
                return;
            }
            
            // Create the focus visualization
            createFocusVisualization(data, 'focus-visualization');
            
            // Update the entity info panel
            updateEntityInfoPanel(data.entity);
            
            // Show the entity info panel
            document.getElementById('entity-info-panel').classList.add('active');
        })
        .catch(error => {
            console.error('Error fetching entity focus data:', error);
            document.getElementById('focus-visualization').innerHTML = 
                `<div class="error">Error loading data: ${error.message}</div>`;
        });
}

// Function to update the entity info panel
function updateEntityInfoPanel(entity) {
    const infoPanel = document.getElementById('entity-info');
    
    // Basic entity info
    let html = `
        <h3>${entity.name}</h3>
        <p class="entity-type">${entity.type.charAt(0).toUpperCase() + entity.type.slice(1)}</p>
    `;
    
    // Add dimension scores if available
    if (entity.position && Object.keys(entity.position).length > 0) {
        html += `<h4>Dimension Scores</h4><div class="dimension-scores">`;
        
        for (const [dimension, score] of Object.entries(entity.position)) {
            // Convert score from [0,1] to [-1,1] for display
            const normalizedScore = score * 2 - 1;
            const percentage = ((normalizedScore + 1) / 2 * 100).toFixed(0);
            
            // Determine color based on dimension
            let color;
            if (dimension === 'economic_axis') {
                color = '#e63946'; // Red for economic
            } else if (dimension === 'social_axis') {
                color = '#457b9d'; // Blue for social
            } else if (dimension === 'ecological_axis') {
                color = '#2a9d8f'; // Green for ecological
            } else {
                color = '#aaa'; // Gray for unknown
            }
            
            html += `
                <div class="dimension-score">
                    <div class="dimension-name">${dimension.replace('_axis', '').replace('_', ' ')}</div>
                    <div class="dimension-value">${normalizedScore.toFixed(2)}</div>
                    <div class="dimension-score-bar">
                        <div class="dimension-score-fill" style="width: ${percentage}%; background-color: ${color};"></div>
                    </div>
                </div>
            `;
        }
        
        html += `</div>`;
    }
    
    infoPanel.innerHTML = html;
}

// Function to go back to the galaxy view
function backToGalaxy() {
    // Hide focus view, show galaxy view
    document.getElementById('focus-container').style.display = 'none';
    document.getElementById('galaxy-container').style.display = 'block';
    
    // Hide the entity info panel
    document.getElementById('entity-info-panel').classList.remove('active');
}
