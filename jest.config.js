/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'node',
  roots: ['./tests'],
  moduleNameMapper: {
    // map d3 to a stub so dataset.ts can be imported without a browser
    '^d3$': '<rootDir>/tests/__mocks__/d3.js',
  },
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        module: 'commonjs',
        lib: ['es2015'],
        esModuleInterop: true,
      },
    }],
  },
};
