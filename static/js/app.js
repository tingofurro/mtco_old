class TreeConversationViewer {
    constructor() {
        this.currentPath = {};
        this.treeData = null;
        this.currentIteration = 1;
        this.availableIterations = [];
        this.highlightedNode = null;
        
        this.initializeEventListeners();
        this.loadInitialData();
    }
    
    initializeEventListeners() {
        // Experiment selection
        document.getElementById('experiment-select').addEventListener('change', () => {
            this.loadConversationData();
        });
        
        // Iteration controls
        document.getElementById('iteration-slider').addEventListener('input', (e) => {
            this.currentIteration = parseInt(e.target.value);
            document.getElementById('iteration-value').textContent = this.currentIteration;
            this.loadConversationData();
        });
        
        document.getElementById('prev-iteration').addEventListener('click', () => {
            if (this.currentIteration > Math.min(...this.availableIterations)) {
                const currentIndex = this.availableIterations.indexOf(this.currentIteration);
                if (currentIndex > 0) {
                    this.currentIteration = this.availableIterations[currentIndex - 1];
                    this.updateIterationControls();
                    this.loadConversationData();
                }
            }
        });
        
        document.getElementById('next-iteration').addEventListener('click', () => {
            if (this.currentIteration < Math.max(...this.availableIterations)) {
                const currentIndex = this.availableIterations.indexOf(this.currentIteration);
                if (currentIndex < this.availableIterations.length - 1) {
                    this.currentIteration = this.availableIterations[currentIndex + 1];
                    this.updateIterationControls();
                    this.loadConversationData();
                }
            }
        });
    }
    
    async loadInitialData() {
        // Load data for the first experiment
        await this.loadConversationData();
    }
    
    async loadConversationData() {
        const experiment = document.getElementById('experiment-select').value;
        if (!experiment) return;
        
        this.showLoading();
        this.hideError();
        
        try {
            const response = await fetch(`/get_conversation_data?experiment=${encodeURIComponent(experiment)}&iteration=${this.currentIteration}`);
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
                return;
            }
            
            this.treeData = data.tree_data;
            this.availableIterations = data.iterations;
            this.currentIteration = data.current_iteration;
            
            // Reset current path when loading new data
            this.currentPath = {};
            this.highlightedNode = null;
            
            this.updateIterationControls();
            this.renderConversation();
            this.renderStats(data.stats);
            
        } catch (error) {
            this.showError(`Failed to load conversation data: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    updateIterationControls() {
        const slider = document.getElementById('iteration-slider');
        const value = document.getElementById('iteration-value');
        const prevBtn = document.getElementById('prev-iteration');
        const nextBtn = document.getElementById('next-iteration');
        
        if (this.availableIterations.length > 0) {
            slider.min = Math.min(...this.availableIterations);
            slider.max = Math.max(...this.availableIterations);
            slider.value = this.currentIteration;
            value.textContent = this.currentIteration;
            
            const currentIndex = this.availableIterations.indexOf(this.currentIteration);
            prevBtn.disabled = currentIndex <= 0;
            nextBtn.disabled = currentIndex >= this.availableIterations.length - 1;
        }
    }
    
    findNodeById(nodeId) {
        return this.treeData.nodes[nodeId];
    }
    
    buildCurrentPath() {
        // Build the current conversation path following user selections
        const nodes = this.treeData.nodes;
        const rootNodeId = this.treeData.conversation_flow[0];
        const conversationPath = [];
        
        let currentNode = this.findNodeById(rootNodeId);
        
        while (currentNode) {
            conversationPath.push(currentNode.id);
            
            if (!currentNode.children || currentNode.children.length === 0) {
                break;
            }
            
            // Choose next node based on current path or default to first child
            let nextNodeId = currentNode.children[0];
            if (this.currentPath[currentNode.id]) {
                nextNodeId = this.currentPath[currentNode.id];
            }
            
            currentNode = this.findNodeById(nextNodeId);
        }
        
        return conversationPath;
    }
    
    renderConversation() {
        if (!this.treeData) return;
        
        const container = document.getElementById('conversation-container');
        
        // Create conversation header
        const header = document.createElement('div');
        header.className = 'conversation-header';
        header.innerHTML = `<h1 class="conversation-title">Tree Conversation Viewer (${this.treeData.conv_id})</h1>`;
        
        // Create messages container
        const messagesContainer = document.createElement('div');
        messagesContainer.className = 'messages-container';
        
        // Render conversation flow
        this.renderConversationFlow(messagesContainer);
        

        
        // Replace container content
        container.innerHTML = '';
        container.appendChild(header);
        container.appendChild(messagesContainer);
    }
    
    renderConversationFlow(container) {
        // Get the current path through the tree
        const currentPath = this.buildCurrentPath();
        
        currentPath.forEach(nodeId => {
            const node = this.findNodeById(nodeId);
            if (!node) return;
            
            const messageElement = this.createMessageElement(node);
            container.appendChild(messageElement);
        });
    }
    
    createMessageElement(node) {
        const messageDiv = document.createElement('div');
        
        if (node.role === 'user') {
            messageDiv.className = 'message user';
            
            let content = this.escapeHtml(node.content || '');
            content += `\n\n*${node.timestamp_formatted}*`;
            
            // Add shard info if available
            if (node.shard_id && node.shard_id !== -1) {
                content += `\n\n<span style="color: blue; font-size: 0.9em;">🧩 Shard revealed: ${node.shard_id}</span>`;
            }
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <span>Node: ${node.id}</span>
                </div>
                <div class="message-content user-content">${content}</div>
            `;
            
        } else if (node.role === 'assistant') {
            const isHighlighted = this.highlightedNode === node.id;
            
            if (node.siblings && node.siblings.length > 1) {
                messageDiv.className = 'message assistant';
                if (isHighlighted) {
                    messageDiv.classList.add('highlighted');
                }
                
                // Create navigation container
                const navContainer = document.createElement('div');
                navContainer.className = 'navigation-container';
                
                // Left arrow
                const leftArrow = document.createElement('button');
                leftArrow.className = 'nav-arrow';
                leftArrow.textContent = '←';
                leftArrow.addEventListener('click', () => this.navigateVariant(node, -1));
                
                // Main message content
                const messageMain = document.createElement('div');
                messageMain.className = 'message-main';
                
                if (isHighlighted) {
                    const highlightDiv = document.createElement('div');
                    highlightDiv.className = 'highlight-indicator';
                    highlightDiv.textContent = '🔸 Recently switched variant 🔸';
                    messageMain.appendChild(highlightDiv);
                }
                
                // Variant info
                const variantInfo = document.createElement('div');
                variantInfo.className = 'variant-info';
                
                let infoText = `Node: ${node.id} | Variant ${node.variant_index + 1}/${node.total_variants}`;
                
                if (node.strategy_name) {
                    infoText += ` | Generation Strategy: ${node.strategy_name}`;
                }
                
                if (node.response_strategy) {
                    infoText += ` | Response Type: ${node.response_strategy}`;
                }
                
                // Add guardrail information
                const guardrailInfo = this.formatGuardrailInfo(node);
                if (guardrailInfo) {
                    infoText += ` | ${guardrailInfo}`;
                }
                
                if (node.score !== undefined) {
                    infoText += ` | Local Score: ${node.score.toFixed(2)}`;
                }
                
                if (node.backpropagated_score !== undefined) {
                    infoText += ` | Backtrack Score: ${node.backpropagated_score.toFixed(2)}`;
                }
                
                // Add advantage information if available
                if (node.advantage !== undefined) {
                    infoText += ` | Advantage: ${node.advantage.toFixed(3)}`;
                }
                
                // Add logprob information if available
                if (node.logprobs !== undefined) {
                    infoText += ` | LogProb: ${node.logprobs.toFixed(2)}`;
                }
                
                if (node.num_tokens !== undefined) {
                    infoText += ` | Tokens: ${node.num_tokens}`;
                }
                
                if (node.normalized_logprob !== undefined) {
                    infoText += ` | Norm LP: ${node.normalized_logprob.toFixed(3)}`;
                }
                
                variantInfo.innerHTML = infoText;
                messageMain.appendChild(variantInfo);
                
                // Message content - render as Markdown
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                // Parse the main content as Markdown
                let mainContent = node.content || '';
                let renderedContent = marked.parse(mainContent);
                
                // Add timestamp
                renderedContent += `<p><em>${node.timestamp_formatted}</em></p>`;
                
                // Add response strategy info
                if (node.response_strategy) {
                    renderedContent += `<p><span style="color: blue; font-size: 0.9em;">🔍 Response classified as: ${node.response_strategy}</span></p>`;
                }
                
                // Add answer evaluation
                if (node.extracted_answer) {
                    renderedContent += `<p><span style="color: blue; font-size: 0.9em;">📝 Extracted answer:</span></p><div class="answer-box">${this.escapeHtml(node.extracted_answer)}</div>`;
                    
                    if (node.is_correct !== undefined) {
                        const icon = node.is_correct ? "✅" : "❌";
                        const text = node.is_correct ? "Correct" : "Incorrect";
                        renderedContent += `<p><span style="color: blue; font-size: 0.9em;">${icon} Answer evaluation: ${text}</span></p>`;
                    } else if (node.score !== undefined) {
                        renderedContent += `<p><span style="color: blue; font-size: 0.9em;">🔢 Answer evaluation score: ${node.score}</span></p>`;
                    }
                }
                
                contentDiv.innerHTML = renderedContent;
                messageMain.appendChild(contentDiv);
                
                // Right arrow
                const rightArrow = document.createElement('button');
                rightArrow.className = 'nav-arrow';
                rightArrow.textContent = '→';
                rightArrow.addEventListener('click', () => this.navigateVariant(node, 1));
                
                navContainer.appendChild(leftArrow);
                navContainer.appendChild(messageMain);
                navContainer.appendChild(rightArrow);
                
                messageDiv.appendChild(navContainer);
                
                if (isHighlighted) {
                    const highlightDiv2 = document.createElement('div');
                    highlightDiv2.className = 'highlight-indicator';
                    highlightDiv2.textContent = '🔸 Recently switched variant 🔸';
                    messageDiv.appendChild(highlightDiv2);
                }
                
            } else {
                // Single variant message - also render as Markdown
                messageDiv.className = 'message assistant';
                
                let mainContent = node.content || '';
                let renderedContent = marked.parse(mainContent);
                renderedContent += `<p><em>${node.timestamp_formatted}</em></p>`;
                
                // Build header info for single variant
                let headerInfo = `Node: ${node.id}`;
                
                if (node.strategy_name) {
                    headerInfo += ` | Generation Strategy: ${node.strategy_name}`;
                }
                
                if (node.response_strategy) {
                    headerInfo += ` | Response Type: ${node.response_strategy}`;
                }
                
                // Add guardrail information
                const guardrailInfo = this.formatGuardrailInfo(node);
                if (guardrailInfo) {
                    headerInfo += ` | ${guardrailInfo}`;
                }
                
                if (node.score !== undefined) {
                    headerInfo += ` | Local Score: ${node.score.toFixed(2)}`;
                }
                
                if (node.backpropagated_score !== undefined) {
                    headerInfo += ` | Backtrack Score: ${node.backpropagated_score.toFixed(2)}`;
                }
                
                // Add advantage information if available
                if (node.advantage !== undefined) {
                    headerInfo += ` | Advantage: ${node.advantage.toFixed(3)}`;
                }
                
                // Add logprob information if available
                if (node.logprobs !== undefined) {
                    headerInfo += ` | LogProb: ${node.logprobs.toFixed(2)}`;
                }
                
                if (node.num_tokens !== undefined) {
                    headerInfo += ` | Tokens: ${node.num_tokens}`;
                }
                
                if (node.normalized_logprob !== undefined) {
                    headerInfo += ` | Norm LP: ${node.normalized_logprob.toFixed(3)}`;
                }
                
                const headerDiv = document.createElement('div');
                headerDiv.className = 'message-header';
                const headerSpan = document.createElement('span');
                headerSpan.innerHTML = headerInfo;
                headerDiv.appendChild(headerSpan);
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = renderedContent;
                
                messageDiv.appendChild(headerDiv);
                messageDiv.appendChild(contentDiv);
            }
        }
        
        return messageDiv;
    }
    
    navigateVariant(node, direction) {
        const siblings = node.siblings;
        const currentIndex = siblings.indexOf(node.id);
        const newIndex = (currentIndex + direction + siblings.length) % siblings.length;
        const newNodeId = siblings[newIndex];
        
        // Update current path
        this.currentPath[node.parent] = newNodeId;
        
        // Set highlight
        this.highlightedNode = newNodeId;
        
        // Re-render conversation
        this.renderConversation();
        
        // Clear highlight after a short delay
        setTimeout(() => {
            this.highlightedNode = null;
        }, 100);
    }
    
    showLoading() {
        document.getElementById('loading').classList.remove('hidden');
    }
    
    hideLoading() {
        document.getElementById('loading').classList.add('hidden');
    }
    
    showError(message) {
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }
    
    hideError() {
        document.getElementById('error').classList.add('hidden');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatGuardrailInfo(node) {
        if (!node || node.role !== 'assistant') return '';
        
        // Only show guardrail info if these fields exist (to avoid showing for older logs)
        if (node.hasOwnProperty('any_enforced_guardrail_triggered') || 
            node.hasOwnProperty('guardrail_repetition_triggered') || 
            node.hasOwnProperty('guardrail_max_length_triggered')) {
            
            const guardrailStatuses = [];
            
            // Check for specific guardrails
            if (node.guardrail_repetition_triggered) {
                guardrailStatuses.push('🔄 Repetition');
            }
            
            if (node.guardrail_max_length_triggered) {
                guardrailStatuses.push('📏 MaxLength');
            }
            
            // If any enforced guardrail is triggered, add a warning indicator
            if (node.any_enforced_guardrail_triggered) {
                return `<span class="guardrail-warning">⚠️ Guardrails: ${guardrailStatuses.join(', ')}</span>`;
            }
            
            // If guardrails were triggered but not enforced, show them in a less prominent way
            if (guardrailStatuses.length > 0) {
                return `<span class="guardrail-info">🛡️ Guardrails: ${guardrailStatuses.join(', ')}</span>`;
            }
            
            // Show "OK" status when no guardrails triggered
            return `<span style="color: #28a745;">✅ Guardrails: OK</span>`;
        }
        
        return '';
    }
    
    renderStats(stats) {
        const statsSection = document.getElementById('stats-section');
        const statsContent = document.getElementById('stats-content');
        
        if (!stats || Object.keys(stats).length === 0) {
            statsSection.classList.add('hidden');
            return;
        }
        
        // Clear previous content
        statsContent.innerHTML = '';
        
        // Helper function to format numbers
        const formatNumber = (value) => {
            if (typeof value === 'number') {
                return value.toFixed(3);
            }
            return value;
        };
        
        // Helper function to create stat item
        const createStatItem = (label, value) => {
            return `
                <div class="stat-item">
                    <span class="stat-label">${label}:</span>
                    <span class="stat-value">${formatNumber(value)}</span>
                </div>
            `;
        };
        
        let html = '';
        
        // Performance Metrics
        if (stats.avg_leaf_node_scores !== undefined || stats.avg_corr_A_LP !== undefined || stats.avg_corr_A_RL !== undefined) {
            html += '<div class="stat-group">';
            html += '<div class="stat-group-title">Performance</div>';
            
            if (stats.avg_leaf_node_scores !== undefined) {
                html += createStatItem('Avg Leaf Score', stats.avg_leaf_node_scores);
            }
            if (stats.avg_corr_A_LP !== undefined) {
                html += createStatItem('Corr(Adv, LogProb)', stats.avg_corr_A_LP);
            }
            if (stats.avg_corr_A_RL !== undefined) {
                html += createStatItem('Corr(Adv, Length)', stats.avg_corr_A_RL);
            }
            
            html += '</div>';
        }
        
        // Response Statistics
        if (stats.avg_response_length !== undefined || stats.avg_depth_rls) {
            html += '<div class="stat-group">';
            html += '<div class="stat-group-title">Response Stats</div>';
            
            if (stats.avg_response_length !== undefined) {
                html += createStatItem('Avg Response Length', stats.avg_response_length);
            }
            
            if (stats.avg_depth_rls) {
                Object.entries(stats.avg_depth_rls).forEach(([depth, avgLength]) => {
                    html += createStatItem(`Depth ${depth} Avg Length`, avgLength);
                });
            }
            
            html += '</div>';
        }
        
        // Timing Information (show top 5 most time-consuming)
        if (stats.timings) {
            html += '<div class="stat-group">';
            html += '<div class="stat-group-title">Timing (seconds)</div>';
            
            // Sort timings by value and show top ones
            const sortedTimings = Object.entries(stats.timings)
                .filter(([key, value]) => key !== 'total_time') // Exclude total_time from the list
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5);
            
            sortedTimings.forEach(([key, value]) => {
                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                html += createStatItem(formattedKey, value);
            });
            
            if (stats.timings.total_time !== undefined) {
                html += createStatItem('Total Time', stats.timings.total_time);
            }
            
            html += '</div>';
        }
        
        // Other metadata
        if (stats.iteration !== undefined) {
            html += '<div class="stat-group">';
            html += '<div class="stat-group-title">Metadata</div>';
            html += createStatItem('Iteration', stats.iteration);
            html += '</div>';
        }
        
        statsContent.innerHTML = html;
        statsSection.classList.remove('hidden');
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TreeConversationViewer();
}); 