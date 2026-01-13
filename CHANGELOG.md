# Changelog

## [1.5.0](https://github.com/panbanda/islands/compare/islands-v1.4.2...islands-v1.5.0) (2026-01-13)


### Features

* sync all indexes by default ([e78e9b3](https://github.com/panbanda/islands/commit/e78e9b37e881a07a9b287fabca4a161d938f025b))
* sync all indexes by default ([981a3a6](https://github.com/panbanda/islands/commit/981a3a6b46c9ee1c491341fed8a84fa7300cf10c))

## [1.4.2](https://github.com/panbanda/islands/compare/islands-v1.4.1...islands-v1.4.2) (2026-01-13)


### Bug Fixes

* correct module paths in embedding provider ([f386cc7](https://github.com/panbanda/islands/commit/f386cc75b4b355311ba27435715d5b33a7a4ba12))
* enable embeddings by default for functional search ([0422815](https://github.com/panbanda/islands/commit/0422815a4e9cb4d8617592cb26f1967f2c3483b1))
* ignore embedding provider doctest ([14b65e2](https://github.com/panbanda/islands/commit/14b65e2e5ee0e4efd8cc5c5a66eeef8d0cf53869))
* include embeddings in default features ([6c3df5a](https://github.com/panbanda/islands/commit/6c3df5ab0f125fda2e4c2ed693e93e030dc608d4))
* resolve clippy warnings exposed by embeddings feature ([b8c5735](https://github.com/panbanda/islands/commit/b8c573549207f191dc8e290f62522a790aa9a45c))

## [1.4.1](https://github.com/panbanda/islands/compare/islands-v1.4.0...islands-v1.4.1) (2026-01-12)


### Bug Fixes

* correct index path structure for metadata persistence ([259a169](https://github.com/panbanda/islands/commit/259a16908f70df0e7be5a0e8bc82bc5e9ecf2dfc))
* correct index path structure for metadata persistence ([a3a7f31](https://github.com/panbanda/islands/commit/a3a7f31dcf92663b73db8abd7b37fb7d6769e257))

## [1.4.0](https://github.com/panbanda/islands/compare/islands-v1.3.1...islands-v1.4.0) (2026-01-12)


### Features

* **indexer:** add workspace management for multi-repo grouping ([bd9d1ac](https://github.com/panbanda/islands/commit/bd9d1acff8b95072439f46f029c5f004d7cc10c7))
* **indexer:** add workspace management for multi-repo grouping ([220a79e](https://github.com/panbanda/islands/commit/220a79e7696e389ae8afd14f28370c5ac0209c68))


### Bug Fixes

* **deps:** update serde ([#33](https://github.com/panbanda/islands/issues/33)) ([7787e08](https://github.com/panbanda/islands/commit/7787e08f792083635a7f3bd6e31963ee7cf8a009))

## [1.3.1](https://github.com/panbanda/islands/compare/islands-v1.3.0...islands-v1.3.1) (2026-01-10)


### Bug Fixes

* support SSH git URLs and improve default paths ([27c296a](https://github.com/panbanda/islands/commit/27c296a151a0eef4ce606bb98d0c7f69df32225a))

## [1.3.0](https://github.com/panbanda/islands/compare/islands-v1.2.0...islands-v1.3.0) (2026-01-09)


### Features

* add progress bars for indexing and file processing ([8edbe20](https://github.com/panbanda/islands/commit/8edbe20bb66ced49f2e3b8fc39b48a2858e1b57d))
* add progress bars for indexing and file processing ([fd6fb02](https://github.com/panbanda/islands/commit/fd6fb02b987e2dc7a4316a56e301124d3e341227))

## [1.2.0](https://github.com/panbanda/islands/compare/islands-v1.1.0...islands-v1.2.0) (2026-01-09)


### Features

* add native Candle embedding provider ([9452cec](https://github.com/panbanda/islands/commit/9452cec59c3106e52e6b9d2237b26af1164f6a9e))
* add native Candle embedding provider ([99b6894](https://github.com/panbanda/islands/commit/99b68947204c132e8943953672c52755597b000b))


### Bug Fixes

* add User-Agent header to HTTP client ([38ef606](https://github.com/panbanda/islands/commit/38ef606c227a2deb7f7199d5cc5cf4b28c3aefa8))

## [1.1.0](https://github.com/panbanda/islands/compare/islands-v1.0.2...islands-v1.1.0) (2026-01-08)


### Features

* comprehensive test coverage and CLI improvements ([822d082](https://github.com/panbanda/islands/commit/822d082fe31fc3bee4c263e14d3fd3a0f6b6a3ee))

## [1.0.2](https://github.com/panbanda/islands/compare/islands-v1.0.1...islands-v1.0.2) (2026-01-06)


### Bug Fixes

* update Rust to 1.92 and optimize Docker build caching ([c74631b](https://github.com/panbanda/islands/commit/c74631b9ac147e88c08ee064f4d48b91bf701d73))
* update Rust to 1.92 and optimize Docker build caching ([6ca271a](https://github.com/panbanda/islands/commit/6ca271a7044470657a720ddd83fa8b2389874821))

## [1.0.1](https://github.com/panbanda/islands/compare/islands-v1.0.0...islands-v1.0.1) (2026-01-06)


### Bug Fixes

* replace let chains with nested if-let for Rust 1.85 compat ([b34f466](https://github.com/panbanda/islands/commit/b34f466f257879f6bf17691b46349cc4845abf94))
* resolve CI errors and update docs for single-crate structure ([03a44b1](https://github.com/panbanda/islands/commit/03a44b1d7e63dd38af838260873ecaa5dc28a625))
* update docker workflow for single-crate and faster PR builds ([4d8b4e1](https://github.com/panbanda/islands/commit/4d8b4e15db5b3c6c28cc62b615ccbc075f5c29f6))
* update Dockerfile for single-crate structure ([8fb3ea6](https://github.com/panbanda/islands/commit/8fb3ea6f20c2b18b533ce545fece32b79d31912b))

## [1.0.0](https://github.com/panbanda/islands/compare/islands-v0.3.0...islands-v1.0.0) (2026-01-06)


### ⚠ BREAKING CHANGES

* Removes docker-compose.yml and raw k8s/ manifests. Users must migrate to Helm-based deployment.

### Features

* migrate to Helm chart with CI/CD for Docker and chart publishing ([cf0b959](https://github.com/panbanda/islands/commit/cf0b959c5899dfd80a055d2dd5d0d31e0fcebb4d))
* migrate to Helm chart with CI/CD for Docker and chart publishing ([3dfb532](https://github.com/panbanda/islands/commit/3dfb532ffa10c285086c0356ea4068c6a8d62d51))


### Bug Fixes

* CI failures for Docker build and Helm lint ([404faa2](https://github.com/panbanda/islands/commit/404faa258e33094cb4a8299dbcd4d3eadf5ce64b))

## [0.3.0](https://github.com/panbanda/islands/compare/islands-v0.2.5...islands-v0.3.0) (2026-01-06)


### Features

* upgrade reqwest to 0.13 ([3b5670d](https://github.com/panbanda/islands/commit/3b5670d4e42a6e328c6f66853ce227f4cfdef370))

## [0.2.5](https://github.com/panbanda/islands/compare/islands-v0.2.4...islands-v0.2.5) (2026-01-06)


### Bug Fixes

* use shasum on macOS (no sha256sum) ([a301047](https://github.com/panbanda/islands/commit/a3010470fb974a0740a93014aa992349362fef63))

## [0.2.4](https://github.com/panbanda/islands/compare/islands-v0.2.3...islands-v0.2.4) (2026-01-06)


### Bug Fixes

* use vendored OpenSSL for cross-compilation ([674f9cf](https://github.com/panbanda/islands/commit/674f9cf30dcb3b4457be33e7cf4309e8ebe721a8))

## [0.2.3](https://github.com/panbanda/islands/compare/islands-v0.2.2...islands-v0.2.3) (2026-01-06)


### Bug Fixes

* install OpenSSL on macOS CI and correct rust-version ([e01e08a](https://github.com/panbanda/islands/commit/e01e08a0c5c3ae5bf5c267da96dd5c43add302ba))

## [0.2.2](https://github.com/panbanda/islands/compare/islands-v0.2.1...islands-v0.2.2) (2026-01-06)


### Bug Fixes

* update macOS runner from retired macos-13 to macos-14 ([a3cdc8f](https://github.com/panbanda/islands/commit/a3cdc8f6a40d840248d5f8c8786bf523967f390f))

## [0.2.1](https://github.com/panbanda/islands/compare/islands-v0.2.0...islands-v0.2.1) (2026-01-06)


### Bug Fixes

* trigger patch release ([214d9af](https://github.com/panbanda/islands/commit/214d9affdb477e2bbcfd3b68a1fa7b7c5ee84ec3))

## [0.2.0](https://github.com/panbanda/islands/compare/islands-v0.1.0...islands-v0.2.0) (2026-01-06)


### Features

* add release-please for automated releases ([2a076c2](https://github.com/panbanda/islands/commit/2a076c2fccf23decc0e6121840aa55a1cdca7566))


### Bug Fixes

* correct nested skills directory structure ([19698ad](https://github.com/panbanda/islands/commit/19698ad52581ed47f1a0023c912dcbd155327d68))
* correct Rust edition and release-please config ([cad98f9](https://github.com/panbanda/islands/commit/cad98f975eb2f09c62693f2e3dab1f60e3a4109b))
