/**
 * Political Latent Space - Hybrid Visualization Module
 * 
 * This module handles the hybrid visualization functionality that combines
 * entity-level and word-level embeddings in the political latent space.
 */

class HybridVisualization {
    constructor(options = {}) {
        // Configuration
        this.options = {
            wordCount: options.wordCount || 50,
            wordSizeMin: options.wordSizeMin || 8,
            wordSizeMax: options.wordSizeMax || 16,
            wordOpacity: options.wordOpacity || 0.7,
            entityMarkerSize: options.entityMarkerSize || 12,
            apiEndpoints: {
                hybridVisualization: options.apiEndpoints?.hybridVisualization || '/api/hybrid-visualization',
                nearestWords: options.apiEndpoints?.nearestWords || '/api/word-embeddings/nearest',
                wordCloud: options.apiEndpoints?.wordCloud || '/api/word-cloud/entity'
            }
        };
        
        // State
        this.selectedEntity = null;
        this.isHybridMode = false;
        this.plotlyInstance = null;
    }
    
    /**
     * Initialize the hybrid visualization
     * @param {string} chartElementId - ID of the DOM element for the chart
     * @param {string} loadingElementId - ID of the DOM element for the loading indicator
     */
    initialize(chartElementId, loadingElementId) {
        this.chartElementId = chartElementId;
        this.loadingElementId = loadingElementId;
        
        // Set up click handler for the plot
        const chartElement = document.getElementById(chartElementId);
        if (chartElement) {
            chartElement.on('plotly_click', this.handlePlotClick.bind(this));
        }
    }
    
    /**
     * Set the selected entity
     * @param {Object} entity - Entity object with type and name properties
     */
    setSelectedEntity(entity) {
        this.selectedEntity = entity;
    }
    
    /**
     * Toggle hybrid visualization mode
     * @param {boolean} enabled - Whether hybrid mode is enabled
     * @returns {boolean} - Whether the mode was successfully changed
     */
    toggleHybridMode(enabled) {
        // Can only enable hybrid mode if an entity is selected
        if (enabled && !this.selectedEntity) {
            return false;
        }
        
        this.isHybridMode = enabled;
        return true;
    }
    
    /**
     * Set the number of words to display in hybrid visualization
     * @param {number} count - Number of words to display
     */
    setWordCount(count) {
        this.options.wordCount = count;
    }
    
    /**
     * Load and display the visualization
     * @param {Function} onComplete - Callback function when visualization is complete
     */
    loadVisualization(onComplete = null) {
        // Show loading indicator
        this.showLoading(true);
        
        let url;
        const params = new URLSearchParams();
        
        if (this.isHybridMode && this.selectedEntity) {
            // Use hybrid visualization endpoint
            url = this.options.apiEndpoints.hybridVisualization;
            params.append('entity_type', this.selectedEntity.type);
            params.append('entity_name', this.selectedEntity.name);
            params.append('num_words', this.options.wordCount);
        } else {
            // Use standard visualization endpoint
            url = '/api/galaxy-data';
            
            if (this.selectedEntity) {
                params.append('entity_type', this.selectedEntity.type);
                params.append('entity_name', this.selectedEntity.name);
            }
        }
        
        // Add params to URL
        if (params.toString()) {
            url += '?' + params.toString();
        }
        
        // Fetch visualization data
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(figure => {
                // Create plot
                Plotly.newPlot(this.chartElementId, figure.data, figure.layout, {responsive: true});
                this.plotlyInstance = document.getElementById(this.chartElementId);
                
                // Hide loading indicator
                this.showLoading(false);
                
                // Call completion callback if provided
                if (onComplete && typeof onComplete === 'function') {
                    onComplete(figure);
                }
            })
            .catch(error => {
                console.error('Error loading visualization:', error);
                this.showLoading(false);
                
                // Display error message in chart area
                const chartElement = document.getElementById(this.chartElementId);
                if (chartElement) {
                    chartElement.innerHTML = `
                        <div style="text-align: center; padding: 50px;">
                            <h3>Error Loading Visualization</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            });
    }
    
    /**
     * Handle click events on the plot
     * @param {Object} eventData - Plotly click event data
     */
    handlePlotClick(eventData) {
        if (!eventData || !eventData.points || eventData.points.length === 0) {
            return;
        }
        
        const point = eventData.points[0];
        const pointData = point.data;
        
        // Check if this is an entity point (not a word)
        if (pointData.name && (pointData.name.includes('Parties') || pointData.name.includes('Politicians'))) {
            const entityType = pointData.name.includes('Parties') ? 'party' : 'politician';
            const entityName = point.text;
            
            // Update selected entity
            this.selectedEntity = { type: entityType, name: entityName };
            
            // Trigger entity selection event
            this.triggerEvent('entitySelected', this.selectedEntity);
            
            // Reload visualization with selected entity
            this.loadVisualization();
        } 
        // Check if this is a word point
        else if (pointData.name === 'Related Words') {
            const word = point.text;
            
            // Trigger word selection event
            this.triggerEvent('wordSelected', { word });
        }
    }
    
    /**
     * Load related words for the selected entity
     * @param {Function} onComplete - Callback function when loading is complete
     */
    loadRelatedWords(onComplete = null) {
        if (!this.selectedEntity) {
            if (onComplete) onComplete([]);
            return;
        }
        
        const url = `${this.options.apiEndpoints.nearestWords}?entity_type=${this.selectedEntity.type}&entity_name=${this.selectedEntity.name}&top_n=${this.options.wordCount}`;
        
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (onComplete && typeof onComplete === 'function') {
                    onComplete(data.nearest_words || []);
                }
            })
            .catch(error => {
                console.error('Error loading related words:', error);
                if (onComplete) onComplete([]);
            });
    }
    
    /**
     * Load word cloud for the selected entity
     * @param {Function} onComplete - Callback function when loading is complete
     */
    loadWordCloud(onComplete = null) {
        if (!this.selectedEntity) {
            if (onComplete) onComplete({});
            return;
        }
        
        const url = `${this.options.apiEndpoints.wordCloud}?entity_type=${this.selectedEntity.type}&entity_name=${this.selectedEntity.name}`;
        
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (onComplete && typeof onComplete === 'function') {
                    onComplete(data.word_cloud || {});
                }
            })
            .catch(error => {
                console.error('Error loading word cloud:', error);
                if (onComplete) onComplete({});
            });
    }
    
    /**
     * Show or hide the loading indicator
     * @param {boolean} show - Whether to show the loading indicator
     */
    showLoading(show) {
        const loadingElement = document.getElementById(this.loadingElementId);
        if (loadingElement) {
            loadingElement.style.display = show ? 'block' : 'none';
        }
    }
    
    /**
     * Trigger a custom event
     * @param {string} eventName - Name of the event
     * @param {Object} data - Event data
     */
    triggerEvent(eventName, data) {
        const event = new CustomEvent(`hybridvis:${eventName}`, { 
            detail: data,
            bubbles: true,
            cancelable: true
        });
        
        document.getElementById(this.chartElementId).dispatchEvent(event);
    }
    
    /**
     * Add event listener for hybrid visualization events
     * @param {string} eventName - Name of the event
     * @param {Function} callback - Event handler function
     */
    on(eventName, callback) {
        document.getElementById(this.chartElementId).addEventListener(
            `hybridvis:${eventName}`,
            event => callback(event.detail)
        );
    }
}

// Export for use in other scripts
window.HybridVisualization = HybridVisualization;
