import pytest
import jax.numpy as jnp
from dmff import NeighborList
import numpy.testing as npt

class TestNeighborList:
    
    @pytest.fixture(scope="class", name='nblist1')
    def test_nblist_init1(self):
        box_vector = jnp.ones(3) * 3
        r_cut = 0.1
        _positions = jnp.linspace(0.5, 0.7, 10)
        positions = jnp.stack([_positions, _positions, _positions], axis=1)
        
        nblist = NeighborList(box_vector, r_cut)
        nblist.allocate(positions)
        yield nblist
        
    @pytest.fixture(scope="class", name='nblist2')
    def test_nblist_init2(self):
        box_vector = jnp.ones(3) * 3
        r_cut = 0.1
        _positions = jnp.linspace(0.5, 0.7, 10)
        positions = jnp.stack([_positions, _positions, _positions], axis=1)
        
        nblist = NeighborList(box_vector[0], r_cut)
        nblist.allocate(positions)
        yield nblist
        
    def test_box_vector_representation(self, nblist1, nblist2):
        npt.assert_allclose(nblist1.nblist.idx, nblist2.nblist.idx)
        
    def test_pairs(self, nblist1):
        
        pairs = nblist1.pairs
        assert pairs.shape == (21, 2)
        
    def test_pair_mask(self, nblist1):
        
        pair, mask = nblist1.pair_mask
        assert mask.shape == (21, )
        
    def test_dr(self, nblist1):
        
        dr = nblist1.dr
        assert dr.shape == (21, 3)
        
    def test_distance(self, nblist1):
        
        assert nblist1.distance.shape == (21, )
