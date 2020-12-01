This model annotates each word or term in a piece of text with a tag representing the entity type, taken from a list of 145  entity tags from the [GENIA Term corpus version 3.02](http://www.geniaproject.org/genia-corpus/term-corpus).

These tags cover 36 types of biological named entities: 
- protein(family_or_group,complex, molecule, subunit, substructure, domain_or_region, other)
- peptide
- amino_acid_monomer
- DNA/RNA(family_or_group, molecule, substructure, domain_or_region, other),- polynucleotide
- nucleotide
- multi_cell
- mono_cell
- virus
- body_part 
- tissue
- cell_type
- cell_component
- cell_line
- other_artificial_source
- lipid
- carbohydrate
- other_organic_compound
- inorganic
- atom
- a tag for 'no entity'

You can refer to the the [GENIA corpus—a semantically annotated corpus
for bio-textmining](https://academic.oup.com/bioinformatics/article/19/suppl_1/i180/227927) for full entity definitions.

The entity types furthermore may be tagged with either a `"B-"`, `"I-"`, `"L-"`, or `"U-"` tag. A `"U-"` tag manifests  only term of a single-term entity. A `"B-"` tag  indicates the first term of a new multi-term entity, while subsequent middle terms in an entity will have an `"I-"` tag and the last term will have the `"L-"` tag. For example, "monocytes" would be tagged as `"U-Cell_Type"` while  "human-immunodeficiency virus type 2" would be tagged as `["B-Virus", "I-Virus", "I-Virus", "I-Virus", "L-Virus"]`.
