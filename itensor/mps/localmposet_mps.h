//
//  localmposet_mps.h
//  
//
//  Created by Mingru Yang on 7/31/17.
//
//

#ifndef __LOCALMPOSET_MPS_H
#define __LOCALMPOSET_MPS_H
#include "itensor/mps/localmpo.h"
#include "itensor/mps/localmposet.h"

namespace itensor {
    
    template <class Tensor>
    class LocalMPOSet_MPS{
    public:
        using LocalMPOSetType = LocalMPOSet<Tensor>;
		using LocalMPOType = LocalMPO<Tensor>;
    private:
        std::vector<MPOt<Tensor>> const* Op_ = nullptr;
        std::vector<MPSt<Tensor>> const* psis_ = nullptr;
        LocalMPOSetType lmpo_;
        std::vector<LocalMPOType> lmps_;
        Real weight_ = 1;
    public:
        LocalMPOSet_MPS() { }
        
        LocalMPOSet_MPS(std::vector<MPOt<Tensor>> const& Op,
                     std::vector<MPSt<Tensor> > const& psis,
                     Args const& args = Args::global());
        
        //
        // Sparse matrix methods
        //
        
        void
        product(Tensor const& phi,
                     Tensor & phip) const;
        
        Real
        expect(Tensor const& phi) const { return lmpo_.expect(phi); }
        
        Tensor deltaRho(Tensor const& AA,
                        Tensor const& comb,
                        Direction dir) const
        { return lmpo_.deltaRho(AA,comb,dir); }
        
        Tensor
        diag() const { return lmpo_.diag(); }
        
        template <class MPSType>
        void
        position(int b, MPSType const& psi);
        
        int
        size() const { return lmpo_.size(); }
        
        explicit
        operator bool() const { return bool(Op_); }
        
        Real
        weight() const { return weight_; }
        void
        weight(Real val) { weight_ = val; }
        
        bool
        doWrite() const { return lmpo_.doWrite(); }
        void
        doWrite(bool val) { lmpo_.doWrite(val); }
        
		int
		numCenter() const { return lmpo_.numCenter(); }
		void
		numCenter(int val) { lmpo_.numCenter(val); }
    };
    
    template <class Tensor>
    inline LocalMPOSet_MPS<Tensor>::
    LocalMPOSet_MPS(std::vector<MPOt<Tensor>> const& Op,
                 std::vector<MPSt<Tensor>> const& psis,
                 Args const& args)
    : Op_(&Op),
    psis_(&psis),
    lmps_(psis.size()),
    weight_(args.getReal("Weight",1))
    {
        lmpo_ = LocalMPOSetType(Op);
        
        for(auto j : range(lmps_.size()))
        {
            lmps_[j] = LocalMPOType(psis[j]);
        }
    }

    template <class Tensor>
    void inline LocalMPOSet_MPS<Tensor>::
    product(Tensor const& phi,
            Tensor & phip) const
    {
        lmpo_.product(phi,phip);
		
        Tensor outer;
        for(auto& M : lmps_)
        {
            M.product(phi,outer);
            outer *= weight_;
            phip += outer;
        }
    }
    
    template <class Tensor>
    template <class MPSType>
    void inline LocalMPOSet_MPS<Tensor>::
    position(int b, const MPSType& psi)
    {
        lmpo_.position(b,psi);
        for(auto& M : lmps_)
        {
            M.position(b,psi);
        }
    }
}


#endif /* localmposet_mps_h */
