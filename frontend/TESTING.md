# Frontend Testing Setup

## Installation

Install the test dependencies:

```bash
cd frontend
npm install
```

This will install Jest, React Testing Library, and all related dependencies specified in `package.json`.

## Running Tests

Once dependencies are installed, you can run the tests:

### Watch mode (for development)
```bash
npm test
```

This will run tests and re-run them as you make changes.

### CI mode with coverage
```bash
npm run test:ci
```

This runs tests once and generates a coverage report in `coverage/` directory.

## Test Files

- `__tests__/components/pages/mutation-lab.test.tsx` - Integration tests for the Mutation Lab component
  - Tests mutation configuration (add, remove, update params)
  - Tests Red-Agent API integration (request/response handling)
  - Tests result rendering (recall display, confidence display, TP counts)
  - Tests session tensor construction (backend provided vs reconstructed)

## Expected Test Output

When you run `npm test`, you should see output like:

```
 PASS  __tests__/components/pages/mutation-lab.test.tsx
  MutationLabPage
    Rendering
      ✓ should render Mutation Lab heading and description
      ✓ should display warning when no session data loaded
      ✓ should render mutation configuration panel when mutations exist
    Mutation Management
      ✓ should display threshold input and tensors badge
      ✓ should show Reconstructed when backend tensors not provided
      ✓ should call addMutation when Add Mutation button clicked
      ...
    Red-Agent Evaluation
      ✓ should disable Run Evaluation button when no mutations
      ✓ should call API with correct request structure including threshold
      ✓ should display recall metrics when backend returns them
      ...
    Session Tensor Construction
      ✓ should construct proper session tensor shape from session summaries
      ✓ should use backend-provided tensors when available

Test Suites: 1 passed, 1 total
Tests:       XX passed, XX total
```

## Troubleshooting

### Tests fail with "Cannot find module"
Make sure you're in the `frontend/` directory and have run `npm install`.

### TypeScript errors in test file
Run `npm install` again to ensure `@types/jest` is installed.

### Port conflicts when running dev + tests
Tests run in jsdom environment so they don't need a server. You can run:
```bash
npm test &  # tests in background
npm run dev # dev server in foreground
```

## Next Steps

The test suite is now in place and ready to:
1. Add more test coverage for other components
2. Integrate into CI/CD pipeline (run `npm run test:ci` in CI)
3. Generate coverage reports for code review
