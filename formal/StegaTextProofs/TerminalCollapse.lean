module

public import StegaTextProofs.LASSelectiveComparison

/-!
A selected terminal group has no strict mass. Exact collapse then resolves
the candidate completely. If its continuation is a single terminal source
path, the updated complete-path belief is a point mass and has zero potential.
-/

@[expose] public section

namespace StegaTextProofs

namespace FiniteLaw

variable {α β : Type*} [Fintype α] [Fintype β]

@[ext] theorem ext_prob {p q : FiniteLaw α} (h : ∀ a, p.prob a = q.prob a) : p = q := by
  cases p
  cases q
  simp_all [funext_iff]

theorem point_mix (a : α) (q : α → FiniteLaw β) : (point a).mix q = q a := by
  apply ext_prob
  intro b
  exact expect_point a (fun c => (q c).prob b)

end FiniteLaw

namespace LASGroup

variable {Exact Strict Leaf : Type*} [Fintype Exact] [Fintype Strict] [Fintype Leaf]

theorem exactMass_one_of_no_strict (g : LASGroup Exact Strict)
    (hstrict : ∀ r, g.prior.prob (.inr r) = 0) : g.exactMass = 1 := by
  simpa only [hstrict, Finset.sum_const_zero, add_zero] using g.mass_partition

theorem collapse_point_of_no_strict (g : LASGroup Exact Strict)
    (hstrict : ∀ r, g.prior.prob (.inr r) = 0) (e : Exact) :
    g.collapse e = FiniteLaw.point (.inl e) := by
  classical
  apply FiniteLaw.ext_prob
  intro c
  cases c with
  | inl c => simp [collapse, FiniteLaw.point, g.exactMass_one_of_no_strict hstrict]
  | inr r => simp [collapse, FiniteLaw.point, hstrict]

theorem collapse_mix_of_no_strict (g : LASGroup Exact Strict)
    (hstrict : ∀ r, g.prior.prob (.inr r) = 0) (e : Exact)
    (channel : Exact ⊕ Strict → FiniteLaw Leaf) :
    (g.collapse e).mix channel = channel (.inl e) := by
  rw [g.collapse_point_of_no_strict hstrict, FiniteLaw.point_mix]

end LASGroup

namespace LASDecisionModel

variable {Branch State : Type*} [Fintype Branch] [Fintype State]
variable {Group : State → Type*} [∀ s, Fintype (Group s)]
variable {Exact Strict : (s : State) → Group s → Type*}
  [∀ s k, Fintype (Exact s k)] [∀ s k, Fintype (Strict s k)]
variable {tree : DecisionTree Branch}

theorem belief_next_of_terminal_group (model : LASDecisionModel tree State Group Exact Strict)
    (s : State) (k : Group s) (e : Exact s k) (l : tree.Leaves)
    (hstrict : ∀ r, (model.selected s k).prior.prob (.inr r) = 0)
    (hresolved : model.continuation s k (.inl e) = FiniteLaw.point l) :
    model.belief (model.next s k e) = FiniteLaw.point l := by
  rw [model.posterior, (model.selected s k).collapse_mix_of_no_strict hstrict, hresolved]

theorem potential_next_zero_of_terminal_group
    (model : LASDecisionModel tree State Group Exact Strict)
    (s : State) (k : Group s) (e : Exact s k) (l : tree.Leaves)
    (hstrict : ∀ r, (model.selected s k).prior.prob (.inr r) = 0)
    (hresolved : model.continuation s k (.inl e) = FiniteLaw.point l) :
    model.potential (model.next s k e) = 0 :=
  model.potential_zero_of_resolved _ l
    (model.belief_next_of_terminal_group s k e l hstrict hresolved)

end LASDecisionModel

end StegaTextProofs
