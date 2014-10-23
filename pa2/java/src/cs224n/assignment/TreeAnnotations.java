package cs224n.assignment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import cs224n.ling.Tree;
import cs224n.ling.Trees;
import cs224n.ling.Trees.MarkovizationAnnotationStripper;
import cs224n.util.Filter;

/**
 * Class which contains code for annotating and binarizing trees for
 * the parser's use, and debinarizing and unannotating them for
 * scoring.
 */
public class TreeAnnotations {

	public static Tree<String> annotateTree(Tree<String> unAnnotatedTree) {
        /*
        We implemented 2nd order vertical, 3rd order vertical, 1st order horizontal, 2nd order horizontal
        markovization. By default, only 2nd order vertical markovization is enabled. To enable other markovizations,
        uncomment the lines below. At most one markovization from each category can be enabled at the same time.
        Note: H1Markovization does not work V3Markovization, and H2Markovization does not work because of sparsity.
        Please refer to our report for more details.
        */

//        V2Markovization(unAnnotatedTree, "");
//        V3Markovization(unAnnotatedTree, "", "");

        H1Markovization(unAnnotatedTree, "");
//        H2Markovization(unAnnotatedTree, "","");


		return binarizeTree(unAnnotatedTree);

	}


    private static void V2Markovization(Tree<String> unAnnotatedTree, String parentLabel) {
        if (unAnnotatedTree.isLeaf()) {
            return;
        }
        List<Tree<String>> children = unAnnotatedTree.getChildren();
        String currentLabel = unAnnotatedTree.getLabel();
        for (Tree<String> child : children) {
            V2Markovization(child, currentLabel);
        }
        if (!parentLabel.isEmpty()) {
            unAnnotatedTree.setLabel(currentLabel + "^" + parentLabel);
        }
    }


    private static void V3Markovization(Tree<String> unAnnotatedTree, String parentLabel, String grandparentLabel) {
        if (unAnnotatedTree.isLeaf()) {
            return;
        }
        List<Tree<String>> children = unAnnotatedTree.getChildren();
        String currentLabel = unAnnotatedTree.getLabel();
        for (Tree<String> child : children) {
            V3Markovization(child, currentLabel, parentLabel);
        }
        String newLabel = currentLabel;
        if (!parentLabel.isEmpty()) {
            newLabel += "^" + parentLabel;
            if (!grandparentLabel.isEmpty()) {
                newLabel += "^" + grandparentLabel;
            }
        }
        unAnnotatedTree.setLabel(newLabel);
    }


    private static void H1Markovization(Tree<String> unAnnotatedTree, String context) {
        if (unAnnotatedTree.isLeaf()) {
            return;
        }

        List<Tree<String>> children = unAnnotatedTree.getChildren();
        String currentLabel = unAnnotatedTree.getLabel();
        String prevSiblingLabel = null;
        for (Tree<String> child : children) {
            String currentChildLabel = child.getLabel();
            if (prevSiblingLabel == null) {
                H1Markovization(child, "");
            } else {
                H1Markovization(child, markovStripper(prevSiblingLabel));
            }
            prevSiblingLabel = currentChildLabel;
        }
        String newLabel = currentLabel;
        if (!context.isEmpty()) {
            newLabel += "^@" + context;
        }
        unAnnotatedTree.setLabel(newLabel);
    }

    private static void H2Markovization(Tree<String> unAnnotatedTree, String context, String context2) {
        if (unAnnotatedTree.isLeaf()) {
            return;
        }

        List<Tree<String>> children = unAnnotatedTree.getChildren();
        String currentLabel = unAnnotatedTree.getLabel();
        String prevSiblingLabel = null;
        String nextPrevSiblingLabel = null;
        for (Tree<String> child : children) {
            String currentChildLabel = child.getLabel();
            if (prevSiblingLabel == null) {
                H2Markovization(child, "","");
            } else if (nextPrevSiblingLabel == null) {
                H2Markovization(child, markovStripper(prevSiblingLabel), "");
            } else {
                H2Markovization(child, markovStripper(prevSiblingLabel), markovStripper(nextPrevSiblingLabel));
            }
            nextPrevSiblingLabel = prevSiblingLabel;
            prevSiblingLabel = currentChildLabel;
        }
        String newLabel = currentLabel;
        if (!context.isEmpty()) {
            newLabel += "^@" + context;
            if (!context2.isEmpty()) {
                newLabel += "^@" + context2;
            }
        }

        unAnnotatedTree.setLabel(newLabel);
    }


 

    private static String markovStripper(String markovizedLabel) {
        String stripped = markovizedLabel;
        int symbolIdx = markovizedLabel.indexOf("^");
        if (symbolIdx > 0) {
            stripped = markovizedLabel.substring(0, symbolIdx);
        }
        return stripped;
    }


	private static Tree<String> binarizeTree(Tree<String> tree) {
		String label = tree.getLabel();
		if (tree.isLeaf())
			return new Tree<String>(label);
		if (tree.getChildren().size() == 1) {
			return new Tree<String>
			(label, 
					Collections.singletonList(binarizeTree(tree.getChildren().get(0))));
		}
		// otherwise, it's a binary-or-more local tree, 
		// so decompose it into a sequence of binary and unary trees.
		String intermediateLabel = "@"+label+"->";
		Tree<String> intermediateTree =
				binarizeTreeHelper(tree, 0, intermediateLabel);
		return new Tree<String>(label, intermediateTree.getChildren());
	}

	private static Tree<String> binarizeTreeHelper(Tree<String> tree,
			int numChildrenGenerated, 
			String intermediateLabel) {
		Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
		List<Tree<String>> children = new ArrayList<Tree<String>>();
		children.add(binarizeTree(leftTree));
		if (numChildrenGenerated < tree.getChildren().size() - 1) {
			Tree<String> rightTree = 
					binarizeTreeHelper(tree, numChildrenGenerated + 1, 
							intermediateLabel + "_" + leftTree.getLabel());
			children.add(rightTree);
		}
		return new Tree<String>(intermediateLabel, children);
	} 

	public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {

		// Remove intermediate nodes (labels beginning with "@"
		// Remove all material on node labels which follow their base symbol
		// (cuts at the leftmost - or ^ character)
		// Examples: a node with label @NP->DT_JJ will be spliced out, 
		// and a node with label NP^S will be reduced to NP

		Tree<String> debinarizedTree =
				Trees.spliceNodes(annotatedTree, new Filter<String>() {
					public boolean accept(String s) {
						return s.startsWith("@");
					}
				});
		Tree<String> unAnnotatedTree = 
				(new Trees.FunctionNodeStripper()).transformTree(debinarizedTree);
    Tree<String> unMarkovizedTree =
        (new Trees.MarkovizationAnnotationStripper()).transformTree(unAnnotatedTree);
		return unMarkovizedTree;
	}
}
