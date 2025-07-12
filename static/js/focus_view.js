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
    
    // Position and add word cloud terms using clustering with improved layout
    const positionedTerms = positionTerms(wordCloud.slice(0, 50), width, height);
    
    positionedTerms.forEach((term) => {
        // Determine node size based on term weight with improved non-linear scaling
        const size = 5 + Math.pow(term.value, 1.5) * 15; // Non-linear scaling for better differentiation
        
        // Add the node with improved styling for visibility
        nodes.push({
            id: `term-${term.text}`,
            name: term.text,
            type: 'term',
            size: size, // Improved size scaling
            color: getTermColor(term), // Use pre-assigned color from positioning function
            x: term.x,
            y: term.y,
            value: term.value,
            distance: term.distance || 0,
            position: term.position || []
        });
        
        // Add a link from the focus entity to this term with opacity based on similarity
        links.push({
            source: `${focusEntity.type}-${focusEntity.name}`,
            target: `term-${term.text}`,
            value: Math.max(1, term.value * 3), // Thicker links for more important terms
            color: `rgba(200, 200, 200, ${term.value * 0.7})` // More opacity for important terms
        });
    });
    
    // Apply force-directed layout
    applyForceLayout(nodes, links, width, height);
    
    // Store node positions for transitions
    storeNodePositions(nodes);
    
    // Store nodes and links in global variables for filtering
    window.currentVisualizationNodes = nodes;
    window.currentVisualizationLinks = links;
    
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
            size: nodes.map(node => node.type === 'term' ? 12 : 14),
            color: nodes.map(node => node.type === 'term' ? '#ffffff' : '#fff')
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
    
    // Add word cloud legend to the visualization
    addWordCloudLegend(container);
    
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
    const value = term.value || 0;
    if (value >= 0.8) return '#4dd0e1';      // Very High Similarity
    if (value >= 0.6) return '#4db6ac';      // High Similarity
    if (value >= 0.4) return '#81c784';      // Medium Similarity
    if (value >= 0.2) return '#aed581';      // Low Similarity
    return '#dce775';                       // Very Low Similarity
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

function positionTerms(terms, width, height) {
    // Create a more interpretable layout that organizes terms by similarity
    let positioned = [];
    
    // Center of the visualization
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Calculate the maximum radius for the layout
    const maxRadius = Math.min(width, height) * 0.4;
    
    // Sort terms by importance for better placement
    const sortedTerms = [...terms].sort((a, b) => b.value - a.value);
    
    // Group terms by value ranges for better organization
    const valueRanges = [
        { min: 0.8, max: 1.0, radius: 0.2, color: '#4dd0e1' }, // Very high similarity - closest to center, red
        { min: 0.6, max: 0.8, radius: 0.4, color: '#4db6ac' }, // High similarity - orange
        { min: 0.4, max: 0.6, radius: 0.6, color: '#81c784' }, // Medium similarity - yellow
        { min: 0.2, max: 0.4, radius: 0.8, color: '#aed581' }, // Low similarity - green
        { min: 0.0, max: 0.2, radius: 1.0, color: '#dce775' }  // Very low similarity - blue
    ];
    
    // Golden angle for optimal distribution within each group
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    
    // Track terms by value range for positioning
    const termsByRange = valueRanges.map(() => []);
    
    // Group terms by value range
    sortedTerms.forEach(term => {
        for (let i = 0; i < valueRanges.length; i++) {
            const range = valueRanges[i];
            if (term.value >= range.min && term.value < range.max) {
                termsByRange[i].push(term);
                break;
            }
        }
    });
    
    // Position terms by group
    termsByRange.forEach((terms, rangeIndex) => {
        const range = valueRanges[rangeIndex];
        const baseRadius = maxRadius * range.radius;
        
        terms.forEach((term, index) => {
            // Calculate angle based on position within group
            const angle = index * goldenAngle;
            
            // Add some randomness to radius to avoid perfect circles
            const radiusVariation = 0.1; // 10% variation
            const radius = baseRadius * (1 - radiusVariation/2 + Math.random() * radiusVariation);
            
            // Calculate coordinates
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            
            // Ensure positions stay within bounds
            const boundedX = Math.max(50, Math.min(width - 50, x));
            const boundedY = Math.max(50, Math.min(height - 50, y));
            
            // Add the positioned term with color based on value range
            positioned.push({
                ...term,
                x: boundedX,
                y: boundedY,
                color: range.color
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
            <select id="relationship-type-filter" style="width: 100%; padding: 5px; background: #333; color: #fff; border: 1px solid #555;">
                <option value="all">All</option>
                <option value="alliance">Alliance</option>
                <option value="opposition">Opposition</option>
                <option value="neutral">Neutral</option>
            </select>
        </div>
        <div style="margin-top: 15px;">
            <button id="apply-filters-btn" style="width: 100%; padding: 8px; background: #4a78a5; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Apply Filters</button>
        </div>
    `;
    
    // Add to container
    document.getElementById(containerId).parentNode.appendChild(controls);
    
    // Initialize filter controls with event listeners
    initializeFilterControls();
}

// Add event listeners for filter controls
function initializeFilterControls() {
    const entityTypeFilter = document.getElementById('entity-type-filter');
    const similarityThreshold = document.getElementById('similarity-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const relationshipFilter = document.getElementById('relationship-type-filter');
    const applyFiltersBtn = document.getElementById('apply-filters-btn');

    // Update threshold value display
    thresholdValue.textContent = similarityThreshold.value;
    
    // Only update the threshold value display when slider changes, don't apply filters
    similarityThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
    });
    
    // Apply filters only when the button is clicked
    applyFiltersBtn.addEventListener('click', applyFilters);
}

// Function to apply filters to the visualization
function applyFilters() {
    console.log('applyFilters called');
    
    // Get filter elements with error handling
    const entityTypeFilterEl = document.getElementById('entity-type-filter');
    const similarityThresholdEl = document.getElementById('similarity-threshold');
    const relationshipTypeFilterEl = document.getElementById('relationship-type-filter');
    
    // Check if all filter elements exist
    if (!entityTypeFilterEl || !similarityThresholdEl || !relationshipTypeFilterEl) {
        console.error('Filter elements not found:', { 
            entityTypeFilter: !!entityTypeFilterEl,
            similarityThreshold: !!similarityThresholdEl,
            relationshipTypeFilter: !!relationshipTypeFilterEl
        });
        return;
    }
    
    // Get filter values
    const entityTypeFilter = entityTypeFilterEl.value;
    const similarityThreshold = parseFloat(similarityThresholdEl.value);
    const relationshipTypeFilter = relationshipTypeFilterEl.value;
    
    console.log('Filter values:', { entityTypeFilter, similarityThreshold, relationshipTypeFilter });
    
    // Get all nodes and links
    const allNodes = window.currentVisualizationNodes;
    const allLinks = window.currentVisualizationLinks;
    
    console.log('Global data available:', { 
        nodesAvailable: !!allNodes, 
        linksAvailable: !!allLinks,
        nodeCount: allNodes ? allNodes.length : 0,
        linkCount: allLinks ? allLinks.length : 0
    });
    
    // Get the focus visualization container
    const container = document.getElementById('focus-visualization');
    if (!container) {
        console.error('Focus visualization container not found');
        return;
    }
    
    // Check if we have data
    if (!allNodes || !allNodes.length) {
        console.error('No nodes data available for filtering');
        return;
    }
    
    // Find the focus entity (the fixed node)
    const focusEntity = allNodes.find(node => node.fixed === true) || { type: 'entity', name: 'Focus' };
    console.log('Focus entity:', focusEntity);
    
    // Filter nodes based on entity type and similarity threshold
    const visibleNodes = allNodes.filter(node => {
        // Always show the focus entity
        if (node.fixed === true) return true;
        
        // Filter by entity type if specified
        if (entityTypeFilter !== 'all' && node.type !== entityTypeFilter) return false;
        
        // For non-term entities, apply similarity threshold
        if (node.type !== 'term' && node.similarity !== undefined && node.similarity < similarityThreshold) return false;
        
        // Always show terms unless specifically filtered out by entity type
        return true;
    });
    
    // Debug node filtering
    console.log('Node filtering details:', {
        totalNodes: allNodes.length,
        visibleNodes: visibleNodes.length,
        termNodes: visibleNodes.filter(n => n.type === 'term').length,
        entityTypeFilter: entityTypeFilter,
        similarityThreshold: similarityThreshold
    });
    
    console.log('Filtered nodes:', { 
        visibleNodeCount: visibleNodes.length,
        nodeTypes: [...new Set(visibleNodes.map(n => n.type))]
    });
    
    // Get visible node IDs for link filtering
    const visibleNodeIds = visibleNodes.map(node => node.id);
    
    // Filter links based on criteria
    const visibleLinks = allLinks.filter(link => {
        // Check if both source and target nodes are visible
        if (!visibleNodeIds.includes(link.source) || !visibleNodeIds.includes(link.target)) {
            return false;
        }
        
        // Filter by relationship type for non-term links
        if (!link.target.startsWith('term-') && 
            relationshipTypeFilter !== 'all' && 
            link.relationshipType !== relationshipTypeFilter) {
            return false;
        }
        
        return true;
    });
    
    // Create updated trace for nodes
    const trace = {
        x: visibleNodes.map(node => node.x),
        y: visibleNodes.map(node => node.y),
        mode: 'markers+text',
        marker: {
            size: visibleNodes.map(node => node.size),
            color: visibleNodes.map(node => node.color),
            line: { width: 1, color: '#333' }
        },
        text: visibleNodes.map(node => node.name),
        textposition: 'bottom center',
        textfont: {
            family: 'Arial',
            size: visibleNodes.map(node => node.type === 'term' ? 12 : 14),
            color: visibleNodes.map(node => node.type === 'term' ? '#ffffff' : '#fff')
        },
        hoverinfo: 'text',
        hovertext: visibleNodes.map(node => {
            if (node.type === 'movement') {
                return `Movement: ${node.name}${formatPositionForHover(node.position)}`;
            } else if (node.type === 'politician') {
                return `Politician: ${node.name}<br>Movement: ${node.movement || 'Unknown'}${formatPositionForHover(node.position)}`;
            } else {
                return `Term: ${node.name}<br>Weight: ${node.value ? node.value.toFixed(2) : 'N/A'}${node.source ? '<br>Source: ' + node.source : ''}`;
            }
        }),
        ids: visibleNodes.map(node => node.id)
    };
    
    // Create edges as shapes
    const shapes = visibleLinks.map(link => {
        const sourceNode = visibleNodes.find(node => node.id === link.source);
        const targetNode = visibleNodes.find(node => node.id === link.target);
        
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
    const width = container.clientWidth;
    const height = container.clientHeight;
    
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
    
    // Create a new layout with all necessary settings
    const layout = {
        title: {
            text: `${focusEntity.type}: ${focusEntity.name}`,
            font: {
                color: '#ffffff',
                size: 18
            }
        },
        showlegend: false,
        hovermode: 'closest',
        shapes: shapes,
        // Explicitly set background colors
        paper_bgcolor: '#1a1a1a',
        plot_bgcolor: '#1a1a1a',
    };
    
    // Update the plot
    console.log('Updating plot with:', { 
        tracePointCount: trace.x.length,
        layoutShapesCount: layout.shapes.length,
        containerId: 'focus-visualization'
    });
    
    // Debug trace data
    console.log('First 3 nodes in trace:', trace.x.slice(0, 3).map((x, i) => ({
        x: x,
        y: trace.y[i],
        name: trace.text[i],
        color: trace.marker.color[i]
    })));
    
    try {
        // First, clear the container completely
        container.innerHTML = '';
        
        console.log('Container cleared, creating new plot');
        
        // Create a completely new plot
        Plotly.newPlot('focus-visualization', [trace], layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toggleSpikelines'],
            displaylogo: false
        });
        
        console.log('New plot created successfully');
        
        // Re-add click event handler using Plotly's on method
        const myPlot = document.getElementById('focus-visualization');
        myPlot.on('plotly_click', function(data) {
            if (!data || !data.points || !data.points[0]) return;
            
            const pointIndex = data.points[0].pointIndex;
            if (pointIndex >= visibleNodes.length) {
                console.error('Point index out of bounds:', pointIndex, 'visibleNodes length:', visibleNodes.length);
                return;
            }
            
            const clickedNode = visibleNodes[pointIndex];
            console.log('Clicked node:', clickedNode);
            
            // Skip if clicked on the focus entity or a term
            if (clickedNode.fixed === true || clickedNode.type === 'term') {
                return;
            }
            
            // Navigate to the clicked entity
            loadEntityFocusView(clickedNode.type, clickedNode.name);
        });
    } catch (error) {
        console.error('Error creating plot:', error);
        // Show error in container
        container.innerHTML = `<div class="error">Error applying filters: ${error.message}</div>`;
    }
}

// Function to load entity word cloud with improved parameters
function loadEntityWordCloud(entityType, entityName) {
    // Use our improved parameters
    const params = new URLSearchParams({
        entity_type: entityType,
        entity_name: entityName.toLowerCase(), // Use lowercase to match backend keys
        top_n: 100,
        max_distance: 5.0,
        filter_stopwords: true
    });
    
    return fetch(`/api/word-cloud/entity?${params}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading word cloud:', data.message);
                return [];
            }
            return data.word_cloud;
        })
        .catch(error => {
            console.error('Error fetching word cloud:', error);
            return [];
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
    
    // First load entity data
    fetch(`/api/entity-focus?entity_type=${entityType}&entity_name=${encodeURIComponent(entityName)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('focus-visualization').innerHTML = 
                    `<div class="error">Error: ${data.message}</div>`;
                return;
            }
            
            // Then load improved word cloud separately
            loadEntityWordCloud(entityType, entityName)
                .then(wordCloud => {
                    // Combine the data
                    data.word_cloud = wordCloud;
                    
                    // Create the focus visualization with improved word cloud
                    createFocusVisualization(data, 'focus-visualization');
                    
                    // Update the entity info panel
                    updateEntityInfoPanel(data.entity);
                    
                    // Show the entity info panel
                    document.getElementById('entity-info-panel').classList.add('active');
                })
                .catch(error => {
                    console.error('Error loading word cloud:', error);
                    // Still show visualization with fallback word cloud
                    createFocusVisualization(data, 'focus-visualization');
                });
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

// Function to add a legend for the word cloud
function addWordCloudLegend(container) {
    // Remove any existing legend
    const existingLegend = container.querySelector('.word-cloud-legend');
    if (existingLegend) {
        existingLegend.remove();
    }
    
    // Create legend container
    const legend = document.createElement('div');
    legend.className = 'word-cloud-legend';
    
    // Style the legend
    legend.style.position = 'absolute';
    legend.style.top = '20px';
    legend.style.left = '20px';
    legend.style.backgroundColor = 'rgba(40, 40, 40, 0.85)';
    legend.style.padding = '12px';
    legend.style.borderRadius = '8px';
    legend.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
    legend.style.fontSize = '13px';
    legend.style.color = '#f0f0f0';
    legend.style.zIndex = '1000';
    
    // Add legend content
    legend.innerHTML = `
        <h4 style="margin-top: 0; margin-bottom: 10px; font-size: 15px; border-bottom: 1px solid #555; padding-bottom: 5px;">Word Cloud Legend</h4>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #4dd0e1; border-radius: 50%; margin-right: 8px;"></span>
            <span>Very High Similarity (0.8-1.0)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #4db6ac; border-radius: 50%; margin-right: 8px;"></span>
            <span>High Similarity (0.6-0.8)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #81c784; border-radius: 50%; margin-right: 8px;"></span>
            <span>Medium Similarity (0.4-0.6)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #aed581; border-radius: 50%; margin-right: 8px;"></span>
            <span>Low Similarity (0.2-0.4)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #dce775; border-radius: 50%; margin-right: 8px;"></span>
            <span>Very Low Similarity (0.0-0.2)</span>
        </div>
    `;
    
    // Add legend to container
    container.appendChild(legend);
}

// Function to go back to the galaxy view
function backToGalaxy() {
    // Hide focus view, show galaxy view
    document.getElementById('focus-container').style.display = 'none';
    document.getElementById('galaxy-container').style.display = 'block';
    
    // Hide the entity info panel
    document.getElementById('entity-info-panel').classList.remove('active');
    
    // Reload the galaxy visualization
    loadGalaxyVisualization();
}
