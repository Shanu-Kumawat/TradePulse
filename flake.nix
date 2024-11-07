{
  description = "Python environment with streamlit, yfinance, plotly, and prophet";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
    in
    {
      devShell.x86_64-linux = pkgs.mkShell {
        buildInputs = [
          pkgs.python312Packages.streamlit
          pkgs.python312Packages.yfinance
          pkgs.python312Packages.plotly
          pkgs.python312Packages.prophet
        ];

        shellHook = ''
          echo "Python development environment with streamlit, yfinance, plotly, and prophet"
        '';
      };
    };
}
