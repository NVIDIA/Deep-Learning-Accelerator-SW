# Contributing to NVIDIA Deep Learning Accelerator SW

Thank you for your interest in contributing. The following requirements apply
to all contributions to this repository.

## Issue tracking

Before starting an enhancement, bug fix, or other change, open a
[DLA SW issue](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/issues)
describing the proposed work.

## Coding guidelines

- Follow the existing conventions in the files and modules you modify.
- Keep changes focused and avoid unrelated refactoring.
- Include or update documentation and tests when behavior changes.
- Verify that relevant tests pass and that builds complete without warnings or
  errors before submitting a pull request.
- Write clear commit titles in the imperative mood and reference the related
  issue when applicable.

## Signing your work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Pull requests

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the
   [NVIDIA Deep Learning Accelerator SW repository](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW).
2. Create a focused branch in your fork and make the changes.
3. Add or update tests and documentation as appropriate.
4. Open a
   [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
   against this repository and describe the change, its motivation, and how it
   was tested.

Pull requests are merged only after every commit is signed off, review is
complete, and the relevant tests pass.
