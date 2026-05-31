# Project Structure And Hygiene

- Follow a modular approach when generating or modifying code.
- Extend the existing package structure where possible instead of creating ad hoc files.
- Make the smallest change that fully solves the task.
- Do not create unnecessary .md files, scratch scripts, demo files, notebooks, or helper files.
- Only create new files when they are required for the implementation.
- If temporary scripts or files are created for debugging, migration, inspection, or generation, delete them before completing the task.
- Before finishing, review the workspace and remove:
  - temporary files
  - duplicate scripts
  - dead code
  - unused imports
  - unnecessary documentation files
- Keep entry-point files thin; place core logic in dedicated modules.
- Preserve backward compatibility unless the task explicitly allows breaking changes.
- Do not rename or reorganize files unless clearly required.
- Before marking the task complete, verify that only necessary files remain.
