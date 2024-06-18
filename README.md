# ml-common
Common functions and variables shared across projects.

# How-to Guide
How to include this library in a project?
```bash
    cd <project directory>
    git submodule add https://github.com/ml4oncology/ml-common.git common
```

How to clone a project that has a submodule?
```bash
    git clone --recurse-submodules <project URL>
```

How to update a submodule inside a project?
```bash
    cd <project directory>
    git submodule update --init --force --remote <submodule-name>
```

