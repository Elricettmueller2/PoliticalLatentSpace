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
        color: getPoliticalSpectrumColor(focusEntity),
        x: width / 2,
        y: height / 2,
        fixed: true, // Keep the focus entity fixed at the center
        position: focusEntity.position || {}
    });
    
    // Add related entities as nodes
    relatedEntities.forEach((entity, i) => {
        // Calculate initial position based on political dimensions if available
        let x, y;
        
        if (entity.position && entity.position.economic_axis !== undefined && entity.position.social_axis !== undefined) {
            // Position based on political dimensions
            // Convert from [0,1] to [-1,1] range and scale by width/height
            const econPos = entity.position.economic_axis * 2 - 1;
            const socialPos = entity.position.social_axis * 2 - 1;
            
            x = width / 2 + econPos * (width / 3);
            y = height / 2 - socialPos * (height / 3); // Invert Y for correct orientation
        } else {
            // Fallback to circular layout if dimensions not available
            const angle = (i / relatedEntities.length) * 2 * Math.PI;
            const radius = 200; // Distance from center
            x = width / 2 + radius * Math.cos(angle);
            y = height / 2 + radius * Math.sin(angle);
        }
        
        // Determine node size based on similarity and influence
        const influence = entity.influence || 1;
        const size = 10 + entity.similarity * 15 + influence * 5;
        
        // Add the node
        nodes.push({
            id: `${entity.type}-${entity.name}`,
            name: entity.name,
            type: entity.type,
            movement: entity.movement || null,
            size: size,
            color: getPoliticalSpectrumColor(entity),
            x: x,
            y: y,
            similarity: entity.similarity,
            position: entity.position || {}
        });
        
        // Determine relationship type
        const relationshipType = determineRelationshipType(focusEntity, entity);
        
        // Add a link from the focus entity to this entity
        links.push({
            source: `${focusEntity.type}-${focusEntity.name}`,
            target: `${entity.type}-${entity.name}`,
            value: entity.similarity * 5, // Link thickness based on similarity
            color: getRelationshipColor(relationshipType, entity.similarity),
            relationshipType: relationshipType
        });
    });
    
    // Position and add word cloud terms using clustering
    const positionedTerms = positionTerms(wordCloud.slice(0, 50), width, height);
    
    positionedTerms.forEach((term) => {
        // Determine node size based on term weight
        const size = 5 + term.value * 10;
        
        // Add the node
        nodes.push({
            id: `term-${term.text}`,
            name: term.text,
            type: 'term',
            size: size,
            color: getTermColor(term),
            x: term.x,
            y: term.y,
            value: term.value,
            position: term.position || []
        });
        
        // Add a thin link from the focus entity to this term
        links.push({
            source: `${focusEntity.type}-${focusEntity.name}`,
            target: `term-${term.text}`,
            value: 1, // Thin links for terms
            color: 'rgba(200, 200, 200, 0.1)'
        });
    });
    
    // Apply force-directed layout
    applyForceLayout(nodes, links, width, height);
    
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
                return `Movement: ${node.name}${formatPositionForHover(node.position)}`;
            } else if (node.type === 'politician') {
                return `Politician: ${node.name}<br>Movement: ${node.movement || 'Unknown'}${formatPositionForHover(node.position)}`;
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
        
        // Skip if nodes not found
        if (!sourceNode || !targetNode) return null;
        
        // Determine line style based on relationship type
        const lineStyle = getLineStyle(link.relationshipType);
        
        return {
            type: 'line',
            x0: sourceNode.x,
            y0: sourceNode.y,
            x1: targetNode.x,
            y1: targetNode.y,
            line: {
                color: link.color,
                width: link.value,
                dash: lineStyle.dash
            },
            opacity: lineStyle.opacity,
            layer: 'below'
        };
    }).filter(shape => shape !== null);
    
    // Add political dimension axes
    shapes.push(
        // Economic axis (horizontal)
        {
            type: 'line',
            x0: 0,
            y0: height / 2,
            x1: width,
            y1: height / 2,
            line: {
                color: 'rgba(200, 200, 200, 0.3)',
                width: 1,
                dash: 'dash'
            },
            layer: 'below'
        },
        // Social axis (vertical)
        {
            type: 'line',
            x0: width / 2,
            y0: 0,
            x1: width / 2,
            y1: height,
            line: {
                color: 'rgba(200, 200, 200, 0.3)',
                width: 1,
                dash: 'dash'
            },
            layer: 'below'
        }
    );
    
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
        annotations: [
            // Instructions
            {
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
            },
            // Economic axis labels
            {
                text: 'Economic Left',
                showarrow: false,
                x: width * 0.1,
                y: height / 2 - 15,
                font: {
                    family: 'Arial',
                    size: 12,
                    color: '#e63946'
                }
            },
            {
                text: 'Economic Right',
                showarrow: false,
                x: width * 0.9,
                y: height / 2 - 15,
                font: {
                    family: 'Arial',
                    size: 12,
                    color: '#457b9d'
                }
            },
            // Social axis labels
            {
                text: 'Progressive',
                showarrow: false,
                x: width / 2 + 15,
                y: height * 0.1,
                textangle: 90,
                font: {
                    family: 'Arial',
                    size: 12,
                    color: '#2a9d8f'
                }
            },
            {
                text: 'Conservative',
                showarrow: false,
                x: width / 2 + 15,
                y: height * 0.9,
                textangle: 90,
                font: {
                    family: 'Arial',
                    size: 12,
                    color: '#e9c46a'
                }
            }
        ]
    };
    
    // Add relationship indicators
    addRelationshipIndicators(links, nodes, layout);
    
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
        
        // Store current positions for smooth transitions
        storeNodePositions(nodes);
        
        // Navigate to a new focus view for the clicked entity
        loadEntityFocusView(clickedNode.type, clickedNode.name);
    });
    
    // Add filter controls
    addFilterControls(containerId, nodes, links);
    
    // Initialize filter controls after visualization is created
    initializeFilterControls();
}

// Helper function to get political spectrum color based on entity position
function getPoliticalSpectrumColor(entity) {
    if (!entity.position) {
        // Default colors if no position data
        return entity.type === 'movement' ? '#e63946' : '#457b9d';
    }
    
    // Get economic and social positions, defaulting to 0.5 (center) if not available
    const economic = entity.position.economic_axis !== undefined ? entity.position.economic_axis : 0.5;
    const social = entity.position.social_axis !== undefined ? entity.position.social_axis : 0.5;
    
    // Convert from [0,1] to [-1,1] range
    const econ = economic * 2 - 1;
    const soc = social * 2 - 1;
    
    // Left-wing economic (red) to right-wing economic (blue)
    // Progressive social (green tint) to conservative social (yellow tint)
    const r = Math.round(255 * (1 - (econ + 1)/2));
    const g = Math.round(255 * (1 - (soc + 1)/2));
    const b = Math.round(255 * ((econ + 1)/2));
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Helper function to get term color based on its position values
function getTermColor(term) {
    if (!term.position || term.position.length < 3) {
        return 'rgba(200, 200, 200, 0.8)';
    }
    
    // Use position values for RGB components
    const r = Math.abs(term.position[0]) * 255;
    const g = Math.abs(term.position[2]) * 255;
    const b = Math.abs(term.position[1]) * 255;
    
    return `rgba(${r}, ${g}, ${b}, 0.8)`;
}

// Helper function to determine relationship type between entities
function determineRelationshipType(entity1, entity2) {
    if (!entity1.position || !entity2.position) {
        return 'unknown';
    }
    
    // Calculate similarity in political dimensions
    let similarity = 0;
    let dimensions = 0;
    
    for (const dim in entity1.position) {
        if (entity2.position[dim] !== undefined) {
            similarity += 1 - Math.abs(entity1.position[dim] - entity2.position[dim]);
            dimensions++;
        }
    }
    
    const avgSimilarity = dimensions > 0 ? similarity / dimensions : 0;
    
    // Determine relationship type based on similarity
    if (avgSimilarity > 0.7) {
        return 'alliance';
    } else if (avgSimilarity < 0.3) {
        return 'opposition';
    } else {
        return 'neutral';
    }
}

// Helper function to get relationship color
function getRelationshipColor(relationshipType, similarity) {
    switch (relationshipType) {
        case 'alliance':
            return `rgba(100, 200, 100, ${0.3 + similarity * 0.5})`;
        case 'opposition':
            return `rgba(200, 100, 100, ${0.3 + similarity * 0.5})`;
        case 'neutral':
            return `rgba(200, 200, 100, ${0.3 + similarity * 0.5})`;
        default:
            return `rgba(150, 150, 150, ${0.3 + similarity * 0.5})`;
    }
}

// Helper function to get line style based on relationship type
function getLineStyle(relationshipType) {
    switch (relationshipType) {
        case 'alliance':
            return { dash: 'solid', opacity: 0.8 };
        case 'opposition':
            return { dash: 'dash', opacity: 0.7 };
        case 'neutral':
            return { dash: 'dot', opacity: 0.6 };
        default:
            return { dash: 'solid', opacity: 0.5 };
    }
}

// Helper function to format position data for hover text
function formatPositionForHover(position) {
    if (!position || Object.keys(position).length === 0) {
        return '';
    }
    
    let text = '<br><br>Political Position:';
    
    if (position.economic_axis !== undefined) {
        const econ = position.economic_axis * 2 - 1;
        text += `<br>Economic: ${econ.toFixed(2)} (${econ < 0 ? 'Left' : 'Right'})`;
    }
    
    if (position.social_axis !== undefined) {
        const social = position.social_axis * 2 - 1;
        text += `<br>Social: ${social.toFixed(2)} (${social < 0 ? 'Conservative' : 'Progressive'})`;
    }
    
    if (position.ecological_axis !== undefined) {
        const eco = position.ecological_axis * 2 - 1;
        text += `<br>Ecological: ${eco.toFixed(2)}`;
    }
    
    return text;
}

// Global variable to store node positions for transitions
let previousNodePositions = {};

// Helper function to store node positions
function storeNodePositions(nodes) {
    nodes.forEach(node => {
        previousNodePositions[node.id] = { x: node.x, y: node.y };
    });
}

// Helper function to position terms using clustering
function positionTerms(terms, width, height) {
    // Group terms by semantic similarity
    const clusters = {};
    
    terms.forEach(term => {
        // Use the first dimension as a simple clustering key if available
        let clusterKey = 0;
        if (term.position && term.position.length > 0) {
            clusterKey = Math.floor(term.position[0] * 5) / 5;
        }
        
        if (!clusters[clusterKey]) clusters[clusterKey] = [];
        clusters[clusterKey].push(term);
    });
    
    // Position terms by cluster
    let positioned = [];
    Object.entries(clusters).forEach(([key, clusterTerms], i) => {
        const angle = (i / Object.keys(clusters).length) * 2 * Math.PI;
        const clusterX = width/2 + Math.cos(angle) * 300;
        const clusterY = height/2 + Math.sin(angle) * 300;
        
        clusterTerms.forEach((term, j) => {
            const radius = 50 + j * 15;
            const termAngle = angle + (j / clusterTerms.length - 0.5) * Math.PI/4;
            
            positioned.push({
                ...term,
                x: clusterX + Math.cos(termAngle) * radius,
                y: clusterY + Math.sin(termAngle) * radius
            });
        });
    });
    
    return positioned;
}

// Helper function to apply force-directed layout
function applyForceLayout(nodes, links, width, height) {
    // This is a simplified simulation of force layout
    // In a real implementation, you would use D3.js force layout
    
    // Apply repulsive forces between nodes
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const node1 = nodes[i];
            const node2 = nodes[j];
            
            // Skip fixed nodes
            if (node1.fixed || node2.fixed) continue;
            
            // Calculate distance
            const dx = node2.x - node1.x;
            const dy = node2.y - node1.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Skip if too far
            if (distance > 200) continue;
            
            // Apply repulsive force
            const force = 1 / Math.max(distance, 10);
            const forceX = dx * force;
            const forceY = dy * force;
            
            // Move nodes apart
            if (!node1.fixed) {
                node1.x -= forceX;
                node1.y -= forceY;
            }
            
            if (!node2.fixed) {
                node2.x += forceX;
                node2.y += forceY;
            }
        }
    }
    
    // Apply attractive forces along links
    links.forEach(link => {
        const sourceNode = nodes.find(node => node.id === link.source);
        const targetNode = nodes.find(node => node.id === link.target);
        
        if (!sourceNode || !targetNode) return;
        
        // Skip if both nodes are fixed
        if (sourceNode.fixed && targetNode.fixed) return;
        
        // Calculate distance
        const dx = targetNode.x - sourceNode.x;
        const dy = targetNode.y - sourceNode.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Skip if too close
        if (distance < 30) return;
        
        // Apply attractive force
        const force = (distance - 100) * 0.01;
        const forceX = dx * force;
        const forceY = dy * force;
        
        // Move nodes together
        if (!sourceNode.fixed) {
            sourceNode.x += forceX;
            sourceNode.y += forceY;
        }
        
        if (!targetNode.fixed) {
            targetNode.x -= forceX;
            targetNode.y -= forceY;
        }
    });
    
    // Keep nodes within bounds
    nodes.forEach(node => {
        if (node.fixed) return;
        
        const padding = 50;
        node.x = Math.max(padding, Math.min(width - padding, node.x));
        node.y = Math.max(padding, Math.min(height - padding, node.y));
    });
}

// Helper function to add relationship indicators
function addRelationshipIndicators(links, nodes, layout) {
    links.forEach(link => {
        if (!link.relationshipType || link.relationshipType === 'unknown') return;
        
        const sourceNode = nodes.find(n => n.id === link.source);
        const targetNode = nodes.find(n => n.id === link.target);
        
        if (!sourceNode || !targetNode) return;
        
        // Calculate midpoint
        const midX = (sourceNode.x + targetNode.x) / 2;
        const midY = (sourceNode.y + targetNode.y) / 2;
        
        // Add indicator based on relationship type
        const indicators = {
            'alliance': '✓',
            'opposition': '✗',
            'neutral': '•'
        };
        
        // Add annotation
        layout.annotations.push({
            text: indicators[link.relationshipType] || '•',
            x: midX,
            y: midY,
            showarrow: false,
            font: { 
                color: link.color,
                size: 14
            }
        });
    });
}

// Helper function to add filter controls
function addFilterControls(containerId, nodes, links) {
    // Check if controls already exist
    const existingControls = document.getElementById('focus-controls');
    if (existingControls) {
        existingControls.remove();
    }
    
    // Create controls container
    const controls = document.createElement('div');
    controls.id = 'focus-controls';
    controls.className = 'focus-controls';
    controls.style.cssText = `
        position: absolute;
        top: 60px;
        right: 10px;
        background-color: rgba(40, 40, 40, 0.8);
        padding: 10px;
        border-radius: 5px;
        z-index: 100;
    `;
    
    // Add filter options
    controls.innerHTML = `
        <div style="margin-bottom: 10px;">
            <label style="color: #ddd; display: block; margin-bottom: 5px;">Filter by type:</label>
            <select id="entity-type-filter" style="width: 100%; padding: 5px; background: #333; color: #fff; border: 1px solid #555;">
                <option value="all">All</option>
                <option value="movement">Movements</option>
                <option value="politician">Politicians</option>
                <option value="term">Terms</option>
            </select>
        </div>
        <div style="margin-bottom: 10px;">
            <label style="color: #ddd; display: block; margin-bottom: 5px;">Min. similarity:</label>
            <input type="range" id="similarity-threshold" min="0" max="1" step="0.1" value="0.0" style="width: 100%;">
            <span id="threshold-value" style="color: #ddd; display: block; text-align: center;">0.0</span>
        </div>
        <div style="margin-bottom: 10px;">
            <label style="color: #ddd; display: block; margin-bottom: 5px;">Relationship type:</label>
            <select id="relationship-filter" style="width: 100%; padding: 5px; background: #333; color: #fff; border: 1px solid #555;">
                <option value="all">All</option>
                <option value="alliance">Alliance</option>
                <option value="opposition">Opposition</option>
                <option value="neutral">Neutral</option>
            </select>
        </div>
    `;
    
    // Add to container
    document.getElementById(containerId).parentNode.appendChild(controls);
}

// Add event listeners for filter controls
function initializeFilterControls() {
    const entityTypeFilter = document.getElementById('entity-type-filter');
    const similarityThreshold = document.getElementById('similarity-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const relationshipFilter = document.getElementById('relationship-filter');

    // Update threshold value display
    thresholdValue.textContent = similarityThreshold.value;
    
    // Add event listeners
    entityTypeFilter.addEventListener('change', applyFilters);
    similarityThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
        applyFilters();
    });
    relationshipFilter.addEventListener('change', applyFilters);
}

// Apply filters to the visualization
function applyFilters() {
    const entityTypeFilter = document.getElementById('entity-type-filter').value;
    const similarityThreshold = parseFloat(document.getElementById('similarity-threshold').value);
    const relationshipFilter = document.getElementById('relationship-filter').value;
    
    // Get all node and link elements
    const nodeElements = document.querySelectorAll('.node');
    const linkElements = document.querySelectorAll('.link');
    
    // Filter nodes based on entity type
    nodeElements.forEach(node => {
        const nodeData = node.__data__;
        let visible = true;
        
        // Filter by entity type
        if (entityTypeFilter !== 'all' && nodeData.type !== entityTypeFilter) {
            visible = false;
        }
        
        // Apply visibility
        node.style.display = visible ? 'block' : 'none';
    });
    
    // Filter links based on similarity threshold and relationship type
    linkElements.forEach(link => {
        const linkData = link.__data__;
        let visible = true;
        
        // Filter by similarity threshold
        if (linkData.similarity < similarityThreshold) {
            visible = false;
        }
        
        // Filter by relationship type
        if (relationshipFilter !== 'all' && linkData.relationship !== relationshipFilter) {
            visible = false;
        }
        
        // Check if source and target nodes are visible
        const sourceNode = document.querySelector(`.node[data-id="${linkData.source}"]`);
        const targetNode = document.querySelector(`.node[data-id="${linkData.target}"]`);
        
        if ((sourceNode && sourceNode.style.display === 'none') || 
            (targetNode && targetNode.style.display === 'none')) {
            visible = false;
        }
        
        // Apply visibility
        link.style.display = visible ? 'block' : 'none';
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
    
    // Add related entities summary if available
    if (entity.related_summary) {
        html += `
            <h4>Key Relationships</h4>
            <div class="related-summary">
                ${entity.related_summary}
            </div>
        `;
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
