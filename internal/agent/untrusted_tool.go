package agent

import (
	"context"

	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/message"
)

type untrustedObservationTool struct {
	inner fantasy.AgentTool
}

func wrapToolsAsUntrustedObservations(tools []fantasy.AgentTool) []fantasy.AgentTool {
	wrapped := make([]fantasy.AgentTool, len(tools))
	for i, tool := range tools {
		if _, ok := tool.(*untrustedObservationTool); ok {
			wrapped[i] = tool
			continue
		}
		wrapped[i] = &untrustedObservationTool{inner: tool}
	}
	return wrapped
}

func (u *untrustedObservationTool) Info() fantasy.ToolInfo {
	return u.inner.Info()
}

func (u *untrustedObservationTool) ProviderOptions() fantasy.ProviderOptions {
	return u.inner.ProviderOptions()
}

func (u *untrustedObservationTool) SetProviderOptions(opts fantasy.ProviderOptions) {
	u.inner.SetProviderOptions(opts)
}

func (u *untrustedObservationTool) Run(ctx context.Context, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
	resp, err := u.inner.Run(ctx, call)
	if resp.Content != "" {
		resp.Content = message.WrapToolOutputAsUntrusted(call.Name, resp.Content)
	}
	return resp, err
}
