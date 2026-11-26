import Lake
open Lake DSL

package «burnnotice-proofs» where
  version := v!"0.1.0"

@[default_target]
lean_lib BurnNotice where
  srcDir := "BurnNotice"