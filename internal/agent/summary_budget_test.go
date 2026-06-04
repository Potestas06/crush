package agent

import (
	"strings"
	"testing"

	"github.com/charmbracelet/crush/internal/message"
	"github.com/stretchr/testify/require"
)

func TestPrepareMessagesForSummarizationCompactsLargeFetchFor16KContext(t *testing.T) {
	t.Parallel()

	largeFetchOutput := strings.Repeat("<div>Microsoft Docs reference content https://learn.microsoft.com/example repeated paragraph.</div>\n", 25000)
	messages := []message.Message{
		{
			Role: message.User,
			Parts: []message.ContentPart{
				message.TextContent{Text: "Please fetch the Microsoft Docs page and explain the relevant API."},
			},
		},
		{
			Role: message.Assistant,
			Parts: []message.ContentPart{
				message.ToolCall{ID: "call_fetch", Name: "fetch", Input: `{"url":"https://learn.microsoft.com/example"}`, Finished: true},
			},
		},
		{
			Role: message.Tool,
			Parts: []message.ContentPart{
				message.ToolResult{ToolCallID: "call_fetch", Name: "fetch", Content: largeFetchOutput},
			},
		},
		{
			Role: message.User,
			Parts: []message.ContentPart{
				message.TextContent{Text: "Continue with the implementation."},
			},
		},
	}

	prepared := prepareMessagesForSummarization(
		messages,
		16_384,
		4096,
		string(summaryPrompt),
		"",
		buildSummaryPrompt(nil),
	)

	require.True(t, prepared.OK, prepared.Reason)
	require.Positive(t, prepared.Budget)
	require.LessOrEqual(t, estimateMessageTokensFromSession(prepared.Messages), prepared.Budget)

	agent := &sessionAgent{}
	require.True(t, fantasyMessagesWithinBudget(prepared.Messages, prepared.Budget, agent, false))

	var compactedToolOutput string
	for _, msg := range prepared.Messages {
		for _, tr := range msg.ToolResults() {
			if tr.ToolCallID == "call_fetch" {
				compactedToolOutput = tr.Content
			}
		}
	}
	require.Contains(t, compactedToolOutput, "Tool output omitted because it exceeded the compaction budget")
	require.Contains(t, compactedToolOutput, "Tool: fetch")
	require.NotContains(t, compactedToolOutput, strings.Repeat("<div>", 50))
}

func TestPrepareMessagesForSummarizationFallsBackWhenBudgetImpossible(t *testing.T) {
	t.Parallel()

	messages := []message.Message{
		{
			Role: message.User,
			Parts: []message.ContentPart{
				message.TextContent{Text: strings.Repeat("current request ", 5000)},
			},
		},
	}

	prepared := prepareMessagesForSummarization(messages, 512, 4096, string(summaryPrompt), buildSummaryPrompt(nil))
	require.False(t, prepared.OK)

	summary := localFallbackSummary(messages, prepared.Reason)
	require.Contains(t, summary, summarySnapshotNotice)
	require.Contains(t, summary, "wait for the next user instruction")
}

func TestBuildSummaryPromptMarksSnapshotAsPassive(t *testing.T) {
	t.Parallel()

	prompt := buildSummaryPrompt(nil)
	require.Contains(t, prompt, summarySnapshotNotice)
	require.Contains(t, prompt, "Do not phrase the summary as an instruction to execute now")
}
