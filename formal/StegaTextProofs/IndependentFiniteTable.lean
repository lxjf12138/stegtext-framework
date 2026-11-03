module

public import StegaTextProofs.FiniteEntropyChannel

/-! A complete independent table, including coordinates not read by an execution. -/

@[expose] public section

namespace StegaTextProofs.FiniteLaw

open scoped BigOperators
attribute [local instance] Classical.propDecidable
variable {ι : Type*} [Fintype ι] {κ : ι → Type*} [∀ i, Fintype (κ i)]

noncomputable def pi (p : ∀ i, FiniteLaw (κ i)) : FiniteLaw (∀ i, κ i) where
  prob f := ∏ i, (p i).prob (f i)
  nonneg f := Finset.prod_nonneg (fun i _ => (p i).nonneg (f i))
  sum_one := by
    classical
    rw [← Fintype.prod_sum]
    simp only [FiniteLaw.sum_one, Finset.prod_const_one]

theorem pi_expect_coordinate (p : ∀ i, FiniteLaw (κ i)) (i : ι) (f : κ i → ℝ) :
    (pi p).expect (fun r => f (r i)) = (p i).expect f := by
  classical
  let w : ∀ j, κ j → ℝ := Function.update (fun j => (p j).prob) i
    (fun a => (p i).prob a * f a)
  have hw : ∀ r : ∀ j, κ j, (∏ j, (p j).prob (r j)) * f (r i) = ∏ j, w j (r j) := by
    intro r
    rw [← Finset.mul_prod_erase Finset.univ (fun j => (p j).prob (r j)) (Finset.mem_univ i),
      ← Finset.mul_prod_erase Finset.univ (fun j => w j (r j)) (Finset.mem_univ i)]
    have he : (∏ j ∈ Finset.univ.erase i, w j (r j)) =
        ∏ j ∈ Finset.univ.erase i, (p j).prob (r j) := by
      apply Finset.prod_congr rfl
      intro j hj
      simp only [w, Function.update_of_ne (Finset.ne_of_mem_erase hj)]
    rw [he]
    simp only [w, Function.update_self]
    ring
  change (∑ r : ∀ j, κ j, (∏ j, (p j).prob (r j)) * f (r i)) = _
  simp only [hw]
  rw [← Fintype.prod_sum]
  calc
    _ = ∑ a, w i a := by
      apply Finset.prod_eq_single i
      · intro j _ hji
        simp only [w, Function.update_of_ne hji, FiniteLaw.sum_one]
      · simp
    _ = _ := by simp only [w, Function.update_self, expect]

end StegaTextProofs.FiniteLaw
