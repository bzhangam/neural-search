# CHANGELOG
All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). See the [CONTRIBUTING guide](./CONTRIBUTING.md#Changelog) for instructions on how to add changelog entries.

## [Unreleased 3.0](https://github.com/opensearch-project/neural-search/compare/2.x...HEAD)
### Features
### Enhancements
- Set neural-search plugin 3.0.0 baseline JDK version to JDK-2 ([#838](https://github.com/opensearch-project/neural-search/pull/838))
### Bug Fixes
- Fix for nested field missing sub embedding field in text embedding processor ([#913](https://github.com/opensearch-project/neural-search/pull/913))
- Correct NeuralQueryBuilder doEquals() and doHashCode().
### Infrastructure
### Documentation
### Maintenance
### Refactoring

## [Unreleased 2.x](https://github.com/opensearch-project/neural-search/compare/2.18...2.x)
### Features
### Enhancements
- Explainability in hybrid query ([#970](https://github.com/opensearch-project/neural-search/pull/970))
- Support new knn query parameter expand_nested ([#1013](https://github.com/opensearch-project/neural-search/pull/1013))
- Implement pruning for neural sparse ingestion pipeline and two phase search processor ([#988](https://github.com/opensearch-project/neural-search/pull/988))
- Support empty string for fields in text embedding processor ([#1041](https://github.com/opensearch-project/neural-search/pull/1041))
### Bug Fixes
-  Address inconsistent scoring in hybrid query results ([#998](https://github.com/opensearch-project/neural-search/pull/998))
### Infrastructure
### Documentation
### Maintenance
### Refactoring
