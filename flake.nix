{
  description = "Theorem proving with LLM + Isabelle development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {};
      };
    in {
      devShells.default = pkgs.mkShellNoCC {
        # Define environment variables here (cleaner than shellHook)
        env = {
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc pkgs.zlib]}:$LD_LIBRARY_PATH";
        };

        buildInputs = with pkgs; [
          # python ecosystem
          python312
          python312Packages.pip
          uv

          # C++ runtime and other libraries (required for numpy and other compiled packages)
          gcc
          stdenv.cc.cc.lib
          zlib

          # build tools for compiling packages like flash-attn
          cmake
          ninja
          pkg-config

          # additional system libraries
          openssl
          libffi
          ncurses

          # container runtime (docker)
          docker
          docker-compose

          # qisabelle requirements
          curl
          brotli # for decompressing AFP heaps

          # development tools
          git
          wget
          jq
          vim

          # basic system tools
          coreutils
          findutils
          bash
          procps
          htop
          gnutar # for tar operations

          # docker support
          docker-compose
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
          export PYTHONPATH=""

          # ensure docker is running and accessible
          echo "checking container runtime..."
          if docker info >/dev/null 2>&1; then
            echo "docker is ready"
          else
            echo "docker might need to be started"
            echo "try: sudo systemctl start docker"
            echo "or: sudo service docker start"
          fi

          # create project structure
          mkdir -p data models logs

          # check docker setup
          echo "checking container runtime..."
          if docker info >/dev/null 2>&1; then
            echo "docker is ready"
          else
            echo "docker might need to be started"
            echo "try: sudo systemctl start docker"
            echo "or: sudo service docker start"
          fi

          # auto-setup qisabelle if script exists
          if [ -f "./setup_qisabelle.sh" ]; then
            echo ""
            echo "running qisabelle setup..."
            bash ./setup_qisabelle.sh
          else
            echo ""
            echo "qisabelle setup script not found"
            echo "create setup_qisabelle.sh in project root for auto-setup"
          fi

          # sync python dependencies
          if [ -f "pyproject.toml" ]; then
            echo ""
            echo "syncing python dependencies..."
            uv sync
          fi

          echo ""
          echo "theorem proving environment ready!"
          echo "project: $(pwd)"
          echo "python: $(python3 --version)"
          echo "uv: $(uv --version)"
          echo "docker: $(docker --version)"
          echo "brotli: $(brotli --version)"
          echo ""
          echo "to start qisabelle server:"
          echo "  cd qisabelle && (sudo) docker-compose up # sudo might be needed"
        '';
      };
    });
}
