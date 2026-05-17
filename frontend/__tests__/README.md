# Frontend Tests

This directory contains unit and integration tests for the CyberShield v2 frontend.

## Running Tests

Install dependencies first (if not already installed):
```bash
npm install
```

Run tests in watch mode:
```bash
npm test
```

Run tests in CI mode with coverage:
```bash
npm run test:ci
```

## Test Structure

Tests are organized by feature:
- `__tests__/components/pages/` - Page-level component tests
- `__tests__/lib/` - Library function tests

## Test Suites

### Mutation Lab Integration Tests (`mutation-lab.test.tsx`)
Comprehensive tests for the Mutation Lab page component covering:
- **Rendering**: Page heading, warnings, mutation panel visibility
- **Mutation Management**: Threshold input, tensors badge, add/remove mutations, parameter updates
- **Red-Agent Evaluation**: API request/response handling, recall/confidence display, TP count display
- **Session Tensor Construction**: Fallback reconstruction from summaries, backend-provided tensor usage

#### Key Test Cases
- ✓ Renders warning when no session data loaded
- ✓ Displays threshold input and tensors badge
- ✓ Calls API with correct request structure including threshold
- ✓ Displays baseline/mutated recall when available
- ✓ Falls back to confidence averages when recall missing
- ✓ Shows TP counts when baseline_true_positives present
- ✓ Constructs proper 20×12 session tensor shape
- ✓ Uses backend-provided tensors when available

## Mocking Strategy

The tests mock:
- **API calls** via `@/lib/api` - prevents actual HTTP requests
- **Zustand store** via `@/lib/store` - isolates component logic
- **Recharts** - avoids rendering issues with chart libraries
- **Next.js routers** - mocks Next.js navigation hooks

This allows fast, reliable tests without external dependencies.

## Coverage Goals

Target coverage:
- Statements: > 80%
- Branches: > 75%
- Functions: > 80%
- Lines: > 80%

Check coverage:
```bash
npm run test:ci
```

Results are written to `coverage/` directory.

## Adding New Tests

1. Create test file in `__tests__/` mirroring source structure
2. Import component and dependencies
3. Mock external dependencies (API, store, etc.)
4. Write test cases using React Testing Library best practices:
   - Query by accessible role/label when possible
   - Simulate user interactions with `userEvent`
   - Use `waitFor` for async operations
5. Run `npm test` to verify

Example test template:
```typescript
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MyComponent from '@/components/MyComponent';

jest.mock('@/lib/api');
jest.mock('@/lib/store');

describe('MyComponent', () => {
    it('should render with correct text', () => {
        render(<MyComponent />);
        expect(screen.getByText('Expected text')).toBeInTheDocument();
    });

    it('should handle user interaction', async () => {
        render(<MyComponent />);
        await userEvent.click(screen.getByRole('button'));
        expect(/* assertion */);
    });

    it('should handle async operations', async () => {
        render(<MyComponent />);
        await waitFor(() => {
            expect(screen.getByText('Loaded')).toBeInTheDocument();
        });
    });
});
```

## Debugging Tests

Run tests with additional debugging:
```bash
npm test -- --verbose
```

Run a single test file:
```bash
npm test mutation-lab.test.tsx
```

Run tests matching a pattern:
```bash
npm test -- -t "should display recall metrics"
```

## CI Integration

In CI/CD pipelines, use:
```bash
npm run test:ci
```

This runs tests once (no watch mode) and generates coverage reports suitable for CI systems.
