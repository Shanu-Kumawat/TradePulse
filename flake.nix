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
          pkgs.python311Packages.streamlit
          pkgs.python311Packages.yfinance
          pkgs.python311Packages.plotly
          pkgs.python311Packages.prophet
          pkgs.python311Packages.tensorflow
          pkgs.python311Packages.scikit-learn
        ];

        shellHook = ''
          echo "Python development environment with streamlit, yfinance, plotly, and prophet"
        '';
      };
    };
}
