# AI Prompt Chain

![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square&logo=typescript)
![AI](https://img.shields.io/badge/AI-LLM-purple?style=flat-square)

Composable prompt chain builder for LLM workflows. Define multi-step prompt pipelines with template interpolation, conditional steps, retries, and output parsing.

## Features

- Fluent chain builder API
- Template interpolation with `{{variables}}`
- Conditional steps with guards
- Built-in output parsers (JSON, lines, number, boolean)
- Retry support per step
- Middleware support
- Provider-agnostic (OpenAI, Anthropic, etc.)

## Usage

```typescript
import { createChain, Parsers } from '@marwantech/ai-prompt-chain';

const chain = createChain(myProvider)
  .set('language', 'TypeScript')
  .step('outline', 'Create an outline for a {{language}} tutorial about: {{input}}')
  .step('draft', 'Write a tutorial based on this outline:\n{{outline}}')
  .step('review', 'Review and improve this tutorial:\n{{draft}}');

const result = await chain.run('building REST APIs');
console.log(result.output);
```

### With parsers and guards

```typescript
const chain = createChain(provider)
  .step('classify', 'Is this text about code? Answer yes/no: {{input}}', {
    parser: Parsers.boolean(),
  })
  .step('explain', 'Explain this code:\n{{input}}', {
    guard: (ctx) => ctx.variables['classify'] === true,
  });
```

## License

MIT
