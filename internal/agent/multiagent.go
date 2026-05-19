package agent

import (
	"sync"
	"time"
)

// AgentInfo holds information about a monitored agent.
type AgentInfo struct {
	ID              string
	Name            string
	LastActivity    time.Time
	DeceptionScore  float64
	TruthScore      float64
	MessageCount    int
}

// InterAgentMessage represents a message between agents.
type InterAgentMessage struct {
	FromAgent   string
	ToAgent     string
	Timestamp   time.Time
	Content     string
	DeceptionScore float64
}

// MultiAgentGovernor tracks and governs inter-agent communication.
type MultiAgentGovernor struct {
	mu              sync.RWMutex
	agents          map[string]*AgentInfo
	messages        []InterAgentMessage
	driftThreshold  float64
	policies        []GovernancePolicy
}

// GovernancePolicy defines a rule for agent-to-agent communication.
type GovernancePolicy struct {
	Name        string
	Description string
	MaxDeceptionScore float64
	Action      string // "block", "log", "alert"
}

// NewMultiAgentGovernor creates a new multi-agent governor.
func NewMultiAgentGovernor(driftThreshold float64) *MultiAgentGovernor {
	return &MultiAgentGovernor{
		agents:         make(map[string]*AgentInfo),
		messages:       make([]InterAgentMessage, 0),
		driftThreshold: driftThreshold,
		policies: []GovernancePolicy{
			{
				Name:        "no_deception_propagation",
				Description: "Block messages with high deception scores",
				MaxDeceptionScore: 0.7,
				Action:      "block",
			},
		},
	}
}

// RegisterAgent registers an agent for monitoring.
func (g *MultiAgentGovernor) RegisterAgent(id, name string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.agents[id] = &AgentInfo{
		ID:           id,
		Name:         name,
		LastActivity: time.Now(),
	}
}

// MonitorMessage scores and evaluates an inter-agent message.
func (g *MultiAgentGovernor) MonitorMessage(from, to, content string, deceptionScore float64) (*InterAgentMessage, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()

	msg := InterAgentMessage{
		FromAgent:      from,
		ToAgent:        to,
		Timestamp:      time.Now(),
		Content:        content,
		DeceptionScore: deceptionScore,
	}

	// Update agent activity
	if agent, ok := g.agents[from]; ok {
		agent.LastActivity = time.Now()
		agent.DeceptionScore = deceptionScore
		agent.MessageCount++
	}
	if agent, ok := g.agents[to]; ok {
		agent.LastActivity = time.Now()
	}

	g.messages = append(g.messages, msg)

	// Check policies
	blocked := false
	for _, p := range g.policies {
		if deceptionScore > p.MaxDeceptionScore && p.Action == "block" {
			blocked = true
			break
		}
	}

	return &msg, !blocked
}

// DetectDriftPropagation checks if deception is propagating between agents.
func (g *MultiAgentGovernor) DetectDriftPropagation() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var alerts []string
	for agentID, agent := range g.agents {
		if agent.DeceptionScore > g.driftThreshold {
			alerts = append(alerts, agentID)
		}
	}
	return alerts
}

// GetAgentInfo returns information about a specific agent.
func (g *MultiAgentGovernor) GetAgentInfo(id string) (*AgentInfo, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	info, ok := g.agents[id]
	return info, ok
}

// GetAgentCount returns the number of registered agents.
func (g *MultiAgentGovernor) GetAgentCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.agents)
}

// GetMessageCount returns the total number of monitored messages.
func (g *MultiAgentGovernor) GetMessageCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.messages)
}

// AddPolicy adds a new governance policy.
func (g *MultiAgentGovernor) AddPolicy(policy GovernancePolicy) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.policies = append(g.policies, policy)
}
